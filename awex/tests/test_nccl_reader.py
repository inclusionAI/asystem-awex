# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

from awex.meta.meta_server import MetaServer, MetaServerClient
from awex.reader import nccl_reader


class _MetaServerClient:
    def __init__(self):
        self.get_calls = []
        self.deleted_keys = []

    def get_object(self, key, timeout=0, default_value=None):
        self.get_calls.append((key, timeout, default_value))
        return True

    def delete_if_exists(self, key):
        self.deleted_keys.append(key)


def test_wait_colocate_write_finished_reads_before_idempotent_cleanup(monkeypatch):
    barrier_calls = []

    monkeypatch.setattr(nccl_reader.device_util, "current_device", lambda: 3)
    monkeypatch.setattr(
        nccl_reader.dist,
        "barrier",
        lambda group=None, device_ids=None: barrier_calls.append((group, device_ids)),
    )

    rank0_meta = _MetaServerClient()
    rank3_meta = _MetaServerClient()

    nccl_reader._wait_colocate_write_finished(rank0_meta, "rank0_key", "pg")
    nccl_reader._wait_colocate_write_finished(rank3_meta, "rank3_key", "pg")

    assert rank0_meta.get_calls == [("rank0_key", 1024**3, None)]
    assert rank3_meta.get_calls == [("rank3_key", 1024**3, None)]
    assert rank0_meta.deleted_keys == ["rank0_key"]
    assert rank3_meta.deleted_keys == ["rank3_key"]
    assert barrier_calls == [
        ("pg", [3]),
        ("pg", [3]),
        ("pg", [3]),
        ("pg", [3]),
    ]


@pytest.mark.parametrize("shared_key", [False, True])
def test_wait_colocate_write_finished_concurrent_readers(monkeypatch, shared_key):
    """Exercise real HTTP reads/deletes with a delayed reader and CPU barriers."""
    num_readers = 4
    barrier = Barrier(num_readers, timeout=5)
    first_read = Event()
    lock = Lock()
    observed = set()
    deleted = set()
    keys = ["shared" if shared_key else f"device_{rank}" for rank in range(num_readers)]
    server = MetaServer("127.0.0.1", 0)
    server.start()
    monkeypatch.setattr(nccl_reader.device_util, "current_device", lambda: 0)
    monkeypatch.setattr(nccl_reader.dist, "barrier", lambda **kwargs: barrier.wait())

    class Client(MetaServerClient):
        def __init__(self, rank):
            super().__init__(server.host, server.port)
            self.rank = rank

        def get_object(self, key, timeout=0, default_value=None):
            if self.rank == num_readers - 1:
                assert first_read.wait(timeout=5)
            value = super().get_object(key, timeout=3, default_value=default_value)
            with lock:
                observed.add(self.rank)
            if self.rank == 0:
                first_read.set()
            return value

        def delete_if_exists(self, key):
            with lock:
                # A destructive first read would strand the delayed reader
                # when several inference processes share a completion key.
                assert len(observed) == num_readers
            super().delete_if_exists(key)
            with lock:
                deleted.add(self.rank)

    def wait(rank):
        try:
            nccl_reader._wait_colocate_write_finished(Client(rank), keys[rank], "pg")
            # No reader can return before every rank has finished cleanup.
            with lock:
                assert len(deleted) == num_readers
        except BaseException:
            barrier.abort()
            raise

    try:
        writer = MetaServerClient(server.host, server.port)
        for key in set(keys):
            writer.put_object(key, True)
        with ThreadPoolExecutor(max_workers=num_readers) as executor:
            futures = [executor.submit(wait, rank) for rank in range(num_readers)]
            for future in futures:
                future.result(timeout=10)
        assert all(not writer.has_key(key) for key in set(keys))
    finally:
        server.stop()
