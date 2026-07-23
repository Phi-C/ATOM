window.BENCHMARK_DATA = {
  "lastUpdate": 1784836804252,
  "repoUrl": "https://github.com/Phi-C/ATOM",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "junyyang-amd",
            "username": "junyyang-amd",
            "email": "junyyang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c451c40de897348557bf1a8b501fda66d2f56565",
          "message": "[atom-sgl-accuracy] Modify model cache mount (#1352)\n\n* Modify model cache mount",
          "timestamp": "2026-06-26T02:52:53Z",
          "url": "https://github.com/Phi-C/ATOM/commit/c451c40de897348557bf1a8b501fda66d2f56565"
        },
        "date": 1782590137619,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2121.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381895 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381895 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381895 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381895 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381895 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7456.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342124 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342124 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.92,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342124 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.97,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342124 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1342124 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3441.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619547 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619547 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619547 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619547 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619547 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5204.31,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936776 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936776 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936776 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936776 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=936776 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6605.89,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1189060 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1189060 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1189060 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1189060 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1189060 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2123.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382304 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382304 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382304 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382304 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382304 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7407.48,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333347 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333347 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.94,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333347 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.98,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333347 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333347 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3433.92,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618105 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618105 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618105 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618105 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618105 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5188.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=933910 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=933910 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=933910 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=933910 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=933910 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6583.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1185072 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1185072 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1185072 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1185072 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1185072 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2091.79,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376523 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376523 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376523 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376523 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376523 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7373.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327222 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327222 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.96,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327222 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 5.02,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327222 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1327222 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3435.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618446 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618446 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618446 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618446 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618446 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5151.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927213 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927213 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927213 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927213 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927213 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6550.52,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1179093 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1179093 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1179093 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1179093 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1179093 Run: https://github.com/Phi-C/ATOM/actions/runs/28298748731"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "junyyang-amd",
            "username": "junyyang-amd",
            "email": "junyyang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c451c40de897348557bf1a8b501fda66d2f56565",
          "message": "[atom-sgl-accuracy] Modify model cache mount (#1352)\n\n* Modify model cache mount",
          "timestamp": "2026-06-26T02:52:53Z",
          "url": "https://github.com/Phi-C/ATOM/commit/c451c40de897348557bf1a8b501fda66d2f56565"
        },
        "date": 1782676497386,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2200.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=396137 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=396137 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=396137 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=396137 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=396137 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7552.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359461 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359461 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.87,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359461 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.93,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359461 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1359461 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3548.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638804 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638804 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638804 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.86,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638804 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=638804 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5212.75,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938295 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938295 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938295 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938295 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938295 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6654.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197860 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197860 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197860 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197860 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197860 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2217.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=399071 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=399071 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=399071 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=399071 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=399071 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7589.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366072 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.06,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366072 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366072 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366072 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366072 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3539.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=637040 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=637040 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=637040 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.9,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=637040 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=637040 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5199.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935936 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935936 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935936 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935936 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935936 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6632.66,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193878 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193878 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193878 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.71,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193878 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193878 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2209.5,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=397710 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=397710 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=397710 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=397710 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=397710 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7527.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354892 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354892 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354892 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354892 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1354892 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3568.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=642300 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=642300 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=642300 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=642300 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=642300 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5290.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952222 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952222 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952222 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952222 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=952222 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6646.86,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1196435 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1196435 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1196435 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.71,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1196435 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1196435 Run: https://github.com/Phi-C/ATOM/actions/runs/28332652066"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1782764787963,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2126.1,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382698 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382698 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382698 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382698 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382698 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7744.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1394065 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.02,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1394065 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.78,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1394065 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.85,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1394065 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1394065 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3527.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634999 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634999 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634999 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634999 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=634999 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5260.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=946952 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=946952 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=946952 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=946952 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=946952 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6761.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1217057 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1217057 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1217057 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.67,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1217057 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1217057 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2156.8,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388224 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388224 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388224 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388224 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=388224 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7717.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1389233 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.03,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1389233 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.8,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1389233 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.85,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1389233 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1389233 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3536.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=636564 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=636564 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=636564 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=636564 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=636564 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5324.35,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=958383 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=958383 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=958383 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=958383 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=958383 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6772.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1218989 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1218989 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.1,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1218989 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.67,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1218989 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1218989 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2173.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=391184 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=391184 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=391184 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=391184 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=391184 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7715.59,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1388806 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.03,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1388806 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1388806 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.84,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1388806 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1388806 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3501.79,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630322 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630322 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630322 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.86,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630322 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=630322 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5296.08,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953295 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953295 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953295 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953295 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=953295 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6655.03,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197906 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197906 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197906 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197906 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1197906 Run: https://github.com/Phi-C/ATOM/actions/runs/28397350821"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1782851191540,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2345.1,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422118 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422118 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422118 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422118 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=422118 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 8787.52,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1581753 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.78,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1581753 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.31,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1581753 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.2,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1581753 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1581753 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3895.7,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=701226 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=701226 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=701226 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=701226 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=701226 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5869.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1056490 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1056490 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.07,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1056490 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1056490 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1056490 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7582.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1364788 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.02,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1364788 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.86,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1364788 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1364788 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1364788 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2336.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420553 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420553 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420553 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420553 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420553 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 8874.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1597390 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1597390 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.28,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1597390 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.15,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1597390 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1597390 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3877.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697963 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697963 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697963 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697963 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697963 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5836.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1050542 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1050542 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.08,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1050542 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.38,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1050542 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1050542 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7551.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1359248 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.02,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1359248 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1359248 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.35,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1359248 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1359248 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2340.4,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421272 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421272 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421272 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421272 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421272 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 8820.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1587738 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1587738 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.29,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1587738 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1587738 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1587738 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3883.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699092 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699092 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699092 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699092 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699092 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5853.26,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053586 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053586 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053586 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053586 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1053586 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7577.19,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1363895 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.02,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1363895 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.86,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1363895 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.35,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1363895 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1363895 Run: https://github.com/Phi-C/ATOM/actions/runs/28470437142"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1783108641722,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2093.89,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376901 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376901 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376901 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376901 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376901 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7433.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338040 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338040 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338040 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.01,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338040 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338040 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3463.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623437 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623437 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623437 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623437 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623437 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5168.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930396 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930396 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930396 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930396 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930396 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6733.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1212015 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1212015 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1212015 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.69,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1212015 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1212015 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2141.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7124.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1282395 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1282395 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 4.13,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1282395 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.26,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1282395 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1282395 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3468.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624299 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624299 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624299 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624299 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624299 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5213.35,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=938403 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6425.92,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1156665 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1156665 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.21,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1156665 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.8,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1156665 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1156665 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2082.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374876 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374876 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374876 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374876 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=374876 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7501.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350212 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350212 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.89,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350212 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.96,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350212 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350212 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3465.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623711 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623711 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623711 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623711 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=623711 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5275.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=949613 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=949613 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=949613 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=949613 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=949613 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6613.99,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1190519 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1190519 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1190519 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1190519 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1190519 Run: https://github.com/Phi-C/ATOM/actions/runs/28678683386"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1783281357815,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2256.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406090 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406090 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406090 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406090 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=406090 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7594.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366937 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.06,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366937 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.85,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366937 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366937 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1366937 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3628.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653078 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653078 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653078 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653078 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653078 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5375.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=967551 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=967551 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=967551 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=967551 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=967551 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6695.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1205272 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1205272 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.13,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1205272 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.7,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1205272 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1205272 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2253.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405569 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405569 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405569 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405569 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405569 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7557.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1360318 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1360318 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.86,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1360318 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1360318 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1360318 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3603.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=648543 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=648543 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=648543 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.84,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=648543 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=648543 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5329.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=959236 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=959236 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=959236 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=959236 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=959236 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6683.01,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1202941 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1202941 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.13,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1202941 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.7,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1202941 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1202941 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2265.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407773 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407773 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407773 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407773 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=407773 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7609.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1369636 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1369636 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.86,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1369636 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1369636 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1369636 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3629.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653396 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653396 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653396 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653396 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=653396 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5398.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=971698 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=971698 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=971698 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=971698 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=971698 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6712.36,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208224 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208224 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208224 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.71,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208224 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1208224 Run: https://github.com/Phi-C/ATOM/actions/runs/28751406534"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1783369743938,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2198.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=395756 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=395756 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=395756 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=395756 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=395756 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7383.2,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1328976 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1328976 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.96,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1328976 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1328976 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1328976 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3605.95,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=649071 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=649071 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=649071 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=649071 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=649071 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5217.56,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=939160 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=939160 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=939160 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=939160 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=939160 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6541.55,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1177479 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1177479 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1177479 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1177479 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1177479 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2228.03,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=401045 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=401045 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=401045 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=401045 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=401045 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7109.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1279772 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1279772 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 4.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1279772 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.91,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1279772 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1279772 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3589.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646193 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646193 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646193 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.9,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646193 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646193 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5009.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=901793 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=901793 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=901793 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 3.78,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=901793 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=901793 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6580.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184492 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184492 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184492 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.81,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184492 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184492 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2138.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384844 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384844 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384844 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 3.31,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384844 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=384844 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 6338.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1141003 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1141003 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 4.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1141003 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 6.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1141003 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1141003 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3553.35,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=639603 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=639603 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=639603 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=639603 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=639603 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 4843.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=871744 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=871744 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.86,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=871744 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 4.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=871744 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=871744 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 5869.61,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1056530 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.31,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1056530 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1056530 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 3.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1056530 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1056530 Run: https://github.com/Phi-C/ATOM/actions/runs/28817629920"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1783541083634,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2744.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=494009 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=494009 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=494009 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=494009 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=494009 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 9566.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1721993 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.63,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1721993 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.05,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1721993 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.83,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1721993 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1721993 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 4506.25,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=811125 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=811125 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=811125 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=811125 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=811125 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 6648.2,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1196676 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1196676 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.94,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1196676 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1196676 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1196676 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 8330.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1499483 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.93,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1499483 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1499483 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1499483 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1499483 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2710.88,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487959 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487959 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487959 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487959 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487959 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 9381.67,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1688701 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1688701 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.1,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1688701 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.9,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1688701 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1688701 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 4428.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=797072 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=797072 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=797072 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=797072 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=797072 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 6685.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1203321 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1203321 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1203321 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1203321 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1203321 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 8402.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1512376 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1512376 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1512376 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1512376 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1512376 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2707.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487322 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487322 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487322 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487322 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=487322 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 9591.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1726482 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.62,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1726482 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.04,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1726482 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.83,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1726482 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1726482 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 4499.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=809888 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=809888 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=809888 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=809888 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=809888 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 6618.95,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1191411 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1191411 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1191411 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1191411 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1191411 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 8271.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1488954 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.93,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1488954 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.72,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1488954 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1488954 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1488954 Run: https://github.com/Phi-C/ATOM/actions/runs/28968352189"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1783628402470,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2916.59,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=524986 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=524986 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=524986 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=524986 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=524986 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 10866.2,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1955916 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1955916 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1955916 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1955916 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1955916 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 4810.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=865946 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=865946 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=865946 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=865946 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=865946 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 7235.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1302353 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1302353 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1302353 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1302353 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1302353 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 9217.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1659076 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1659076 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1659076 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.05,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1659076 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1659076 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2906.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=523250 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=523250 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=523250 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=523250 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=523250 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 10810.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1945879 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.44,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1945879 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1945879 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1945879 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1945879 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 4783.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=861097 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=861097 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=861097 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=861097 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=861097 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 7176.45,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1291761 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1291761 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.89,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1291761 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1291761 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1291761 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 9265.27,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667748 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667748 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667748 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.03,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667748 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667748 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2920.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=525692 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=525692 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=525692 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=525692 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=525692 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 10445.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1880191 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1880191 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.84,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1880191 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1880191 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1880191 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 4836.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=870585 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=870585 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=870585 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=870585 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=870585 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 7208.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1297528 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1297528 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.89,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1297528 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1297528 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1297528 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 9265.36,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667765 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667765 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667765 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.98,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667765 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1667765 Run: https://github.com/Phi-C/ATOM/actions/runs/29043828331"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1783714606576,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2713.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=488515 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=488515 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=488515 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=488515 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=488515 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 9389.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1690132 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1690132 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1690132 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.94,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1690132 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1690132 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 4441.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=799547 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=799547 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=799547 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=799547 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=799547 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 6505.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1171056 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1171056 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.96,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1171056 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1171056 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1171056 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 8077.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1453992 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1453992 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1453992 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.22,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1453992 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1453992 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2670.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=480712 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=480712 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=480712 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=480712 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=480712 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 9304.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1674753 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.67,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1674753 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.14,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1674753 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.97,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1674753 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1674753 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 4409.69,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=793745 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=793745 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=793745 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=793745 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=793745 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 6463.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1163364 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1163364 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.96,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1163364 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1163364 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1163364 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 8295.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1493111 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1493111 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.71,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1493111 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1493111 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1493111 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2635.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=474450 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=474450 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=474450 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=474450 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=474450 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 9227.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1661037 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1661037 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.16,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1661037 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.99,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1661037 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1661037 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 4379.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=788352 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=788352 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=788352 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=788352 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=788352 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 6610.14,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189825 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189825 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189825 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189825 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189825 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 8177.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1472028 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1472028 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1472028 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.21,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1472028 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1472028 Run: https://github.com/Phi-C/ATOM/actions/runs/29117222729"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1783885733143,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2122.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382083 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382083 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382083 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382083 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382083 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7427.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336930 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336930 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.93,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336930 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.02,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336930 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1336930 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3406.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613191 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613191 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613191 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613191 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=613191 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5163.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929471 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929471 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.24,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929471 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929471 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929471 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6522.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174021 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174021 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174021 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174021 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1174021 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2095.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377153 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377153 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377153 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377153 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377153 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7228.36,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301105 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301105 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 4.05,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301105 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.15,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301105 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301105 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3412.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=614220 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=614220 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=614220 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=614220 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=614220 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5193.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=934915 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=934915 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=934915 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=934915 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=934915 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6428.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1157048 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1157048 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.21,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1157048 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.8,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1157048 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1157048 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2059.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=370637 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=370637 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=370637 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=370637 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=370637 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7146.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1286394 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1286394 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 4.1,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1286394 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 5.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1286394 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1286394 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3356.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=604202 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=604202 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=604202 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.9,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=604202 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=604202 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5090.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=916314 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=916314 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.25,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=916314 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=916314 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=916314 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6396.8,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1151424 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1151424 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1151424 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.83,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1151424 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1151424 Run: https://github.com/Phi-C/ATOM/actions/runs/29204476670"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Xin Huang",
            "username": "gyohuangxin",
            "email": "Xin.Huang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f696e2bc0abd4e26a79ab0b5377f213a3197aa54",
          "message": "CI: use host network for ATOM benchmark container (#1390)",
          "timestamp": "2026-06-29T02:18:06Z",
          "url": "https://github.com/Phi-C/ATOM/commit/f696e2bc0abd4e26a79ab0b5377f213a3197aa54"
        },
        "date": 1783973321162,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 3387.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=609700 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.28,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=609700 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=609700 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=609700 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=609700 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 13490.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2428339 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2428339 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.2,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2428339 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 2.78,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2428339 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2428339 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5549.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=998915 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=998915 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=998915 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=998915 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=998915 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 8386.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1509542 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1509542 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1509542 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1509542 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1509542 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 11341.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2041512 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2041512 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.29,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2041512 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.64,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2041512 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2041512 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 3334.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=600294 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.28,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=600294 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=600294 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=600294 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=600294 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 13629.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2453239 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.14,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2453239 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2453239 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2453239 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2453239 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5537.19,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=996695 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=996695 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=996695 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=996695 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=996695 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 8555.26,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1539947 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1539947 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1539947 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 0.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1539947 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1539947 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 11298.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2033732 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2033732 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.28,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2033732 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.61,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2033732 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2033732 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 3398.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=611810 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.28,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=611810 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=611810 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=611810 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=611810 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 13007.61,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2341370 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2341370 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.27,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2341370 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 2.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2341370 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2341370 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5479.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=986243 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=986243 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=986243 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=986243 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=986243 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 8572.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1543003 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1543003 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1543003 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 0.92,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1543003 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1543003 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 11322.14,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2037986 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2037986 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.29,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2037986 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.69,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2037986 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2037986 Run: https://github.com/Phi-C/ATOM/actions/runs/29277281075"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zufa.yu@amd.com",
            "name": "zufayu",
            "username": "zufayu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T18:13:21+08:00",
          "tree_id": "d485429ad628948733ebf0d0fdf326968c0de590",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784117491356,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2284.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411124 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411124 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411124 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411124 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=411124 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 8586.59,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1545587 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.82,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1545587 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.38,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1545587 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.29,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1545587 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1545587 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3785.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=681323 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=681323 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=681323 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=681323 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=681323 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5739.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033159 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033159 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033159 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033159 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033159 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7368.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1326261 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1326261 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.92,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1326261 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1326261 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1326261 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2247.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=404480 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=404480 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=404480 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=404480 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=404480 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 8450.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1521066 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.85,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1521066 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1521066 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1521066 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1521066 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3751.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675246 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675246 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.7,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675246 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675246 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=675246 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5740.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033357 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033357 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033357 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033357 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1033357 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7385.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1329331 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1329331 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.91,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1329331 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1329331 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1329331 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2251.36,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405244 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405244 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405244 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405244 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405244 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 8307.52,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1495353 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1495353 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1495353 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1495353 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1495353 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3728.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671190 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671190 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.7,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671190 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671190 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671190 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5720.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1029683 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1029683 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1029683 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.4,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1029683 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1029683 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7357.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1324408 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1324408 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.91,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1324408 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1324408 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1324408 Run: https://github.com/Phi-C/ATOM/actions/runs/29410556517"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784145018862,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2334.75,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420255 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420255 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420255 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420255 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420255 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 8610.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1549829 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.81,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1549829 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.38,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1549829 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.31,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1549829 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1549829 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3859.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694623 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694623 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694623 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.81,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694623 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694623 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5790.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042273 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042273 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042273 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042273 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042273 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7513.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1352420 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1352420 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1352420 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1352420 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1352420 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2304.5,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414810 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414810 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414810 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414810 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=414810 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 8490.67,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1528320 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.84,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1528320 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1528320 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1528320 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1528320 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3801.25,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684225 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684225 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.7,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684225 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684225 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684225 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5611.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1010094 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1010094 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1010094 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.94,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1010094 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1010094 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7337.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1320720 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1320720 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.97,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1320720 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.57,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1320720 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1320720 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2317.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417199 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417199 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417199 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417199 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417199 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7921.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1425917 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.96,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1425917 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 4.01,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1425917 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 5.6,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1425917 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1425917 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3820.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687637 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687637 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687637 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687637 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=687637 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5752.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035389 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035389 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035389 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035389 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035389 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7255.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1306037 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1306037 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.96,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1306037 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1306037 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1306037 Run: https://github.com/Phi-C/ATOM/actions/runs/29442247735"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784231758079,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2314.6,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416628 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416628 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416628 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416628 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=416628 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 8725.3,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570554 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570554 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570554 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.26,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570554 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1570554 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3840.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691363 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691363 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691363 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691363 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=691363 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5756.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036091 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036091 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036091 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036091 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036091 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7471,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1344780 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1344780 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1344780 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.39,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1344780 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1344780 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2323.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418198 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418198 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418198 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418198 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418198 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 8670.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1560607 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.8,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1560607 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.35,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1560607 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.27,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1560607 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1560607 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3823.7,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688266 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688266 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688266 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688266 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=688266 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5757.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036335 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036335 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036335 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036335 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1036335 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7417.01,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1335062 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1335062 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.9,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1335062 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.38,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1335062 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1335062 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2319.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417548 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417548 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417548 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417548 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=417548 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 8661.35,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1559043 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.8,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1559043 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1559043 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.28,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1559043 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1559043 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3836.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690508 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690508 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690508 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690508 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690508 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5754.25,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035765 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035765 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.1,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035765 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035765 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1035765 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7325.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1318562 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1318562 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.93,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1318562 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.44,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1318562 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1318562 Run: https://github.com/Phi-C/ATOM/actions/runs/29525729820"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784318016994,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2609.27,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=469669 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=469669 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=469669 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=469669 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=469669 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 10403.86,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1872694 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1872694 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.84,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1872694 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.6,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1872694 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1872694 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 4373.74,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=787274 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=787274 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=787274 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.7,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=787274 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=787274 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 6609.55,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189719 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189719 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189719 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189719 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1189719 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 8870.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1596620 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.86,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1596620 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.64,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1596620 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1596620 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1596620 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2629.27,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=473269 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=473269 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=473269 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=473269 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=473269 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 10077.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1814036 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1814036 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1814036 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.71,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1814036 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1814036 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 4447.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=800628 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=800628 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=800628 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.68,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=800628 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=800628 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 6601.89,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1188340 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1188340 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1188340 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1188340 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1188340 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 8702.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1566392 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1566392 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.64,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1566392 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.05,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1566392 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1566392 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2545.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=458180 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=458180 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=458180 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=458180 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=458180 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 10165.14,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1829726 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1829726 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1829726 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.61,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1829726 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1829726 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 4346.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=782411 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=782411 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=782411 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.7,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=782411 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=782411 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 6581.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1184590 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1184590 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1184590 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1184590 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1184590 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 8494.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1528980 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.9,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1528980 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1528980 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.09,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1528980 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1528980 Run: https://github.com/Phi-C/ATOM/actions/runs/29605266218"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784403907089,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2890.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=520275 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=520275 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=520275 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=520275 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=520275 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 11530.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2075576 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2075576 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2075576 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.33,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2075576 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2075576 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5126.07,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=922693 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=922693 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=922693 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=922693 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=922693 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 7736.66,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1392599 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1392599 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.81,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1392599 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1392599 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1392599 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 10115.02,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1820704 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1820704 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.44,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1820704 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.85,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1820704 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1820704 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2895.03,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521106 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521106 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521106 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521106 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521106 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 11799.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2123882 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.31,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2123882 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2123882 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2123882 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2123882 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5007.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=901389 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.38,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=901389 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=901389 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=901389 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=901389 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 7607.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1369424 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1369424 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1369424 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.06,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1369424 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1369424 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 10270.6,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1848708 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1848708 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1848708 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1848708 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1848708 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2840.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=511228 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=511228 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=511228 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=511228 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=511228 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 11974.69,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2155444 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.3,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2155444 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2155444 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.11,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2155444 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2155444 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5032.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=905909 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.38,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=905909 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=905909 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=905909 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=905909 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 7550.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1359148 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1359148 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1359148 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1359148 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1359148 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 10103.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1818656 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1818656 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1818656 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.81,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1818656 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1818656 Run: https://github.com/Phi-C/ATOM/actions/runs/29656404391"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784490608101,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2088.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375963 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375963 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375963 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375963 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375963 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7248.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304679 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304679 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 4.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304679 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304679 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1304679 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3438.69,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618964 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618964 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618964 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618964 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=618964 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5109.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919759 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919759 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.24,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919759 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919759 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919759 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6450.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161023 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161023 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.21,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161023 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161023 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1161023 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2099.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377970 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377970 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377970 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377970 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=377970 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7307.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315372 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315372 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.99,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315372 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.07,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315372 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1315372 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3446.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620449 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620449 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620449 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620449 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620449 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5150.33,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927060 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927060 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927060 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927060 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=927060 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6406.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1153220 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1153220 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.22,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1153220 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.8,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1153220 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1153220 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2084.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375222 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375222 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375222 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375222 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=375222 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7494.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1349084 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.09,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1349084 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1349084 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.95,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1349084 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1349084 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3342.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=601629 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=601629 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=601629 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.9,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=601629 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=601629 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5145.23,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926141 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926141 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926141 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926141 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926141 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6601.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188242 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188242 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188242 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188242 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188242 Run: https://github.com/Phi-C/ATOM/actions/runs/29699248645"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784579077522,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2117.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381208 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381208 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381208 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381208 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=381208 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7315.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1316722 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1316722 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.99,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1316722 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1316722 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1316722 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3487.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=627740 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=627740 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=627740 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=627740 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=627740 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5199.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935951 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935951 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935951 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935951 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=935951 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6580.2,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184436 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184436 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184436 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184436 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1184436 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2131.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383650 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383650 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383650 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383650 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383650 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7391.26,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1330426 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.12,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1330426 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.95,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1330426 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 5.02,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1330426 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1330426 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3496.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629403 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629403 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629403 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.86,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629403 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=629403 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5179.65,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=932337 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=932337 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=932337 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=932337 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=932337 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6484.69,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1167245 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1167245 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1167245 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.78,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1167245 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1167245 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2108.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379569 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379569 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379569 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379569 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=379569 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7438.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338953 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.1,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338953 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.92,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338953 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338953 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1338953 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3513.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632509 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632509 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632509 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632509 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=632509 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5235.89,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=942461 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=942461 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=942461 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=942461 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=942461 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6570.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1182697 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1182697 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.16,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1182697 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1182697 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1182697 Run: https://github.com/Phi-C/ATOM/actions/runs/29771631395"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784664219567,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2279.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410240 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410240 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410240 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410240 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410240 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 8700.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1566149 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1566149 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.33,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1566149 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.23,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1566149 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1566149 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3797.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683504 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683504 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683504 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683504 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683504 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5809.35,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045683 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045683 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045683 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.37,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045683 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045683 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 7464.24,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1343564 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1343564 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1343564 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.38,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1343564 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1343564 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2276.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409755 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409755 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409755 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409755 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=409755 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 8711.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1568059 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1568059 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.32,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1568059 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.21,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1568059 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1568059 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3802.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684390 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684390 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684390 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684390 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=684390 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5808.1,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045458 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045458 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.08,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045458 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.38,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045458 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045458 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 7523.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354313 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354313 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354313 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354313 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354313 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2269.92,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=408586 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=408586 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=408586 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=408586 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=408586 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 8734.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1572158 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1572158 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.32,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1572158 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.22,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1572158 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1572158 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3796.66,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683398 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683398 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683398 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683398 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=683398 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5789.04,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042027 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042027 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042027 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042027 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1042027 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 7526.27,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354728 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354728 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354728 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354728 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1354728 Run: https://github.com/Phi-C/ATOM/actions/runs/29859682359"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784750108381,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2899.39,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521891 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.33,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521891 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521891 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521891 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=521891 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 12371.9,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2226942 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.26,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2226942 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.39,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2226942 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.05,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2226942 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2226942 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5264.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=947573 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=947573 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=947573 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=947573 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=947573 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 8048.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1448697 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1448697 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1448697 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1448697 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1448697 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 10652.64,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917475 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917475 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917475 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.68,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917475 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917475 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2862.61,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=515270 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=515270 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=515270 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=515270 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=515270 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 12328.79,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2219183 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.26,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2219183 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2219183 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 2.98,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2219183 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2219183 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5249.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944966 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944966 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944966 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944966 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944966 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 7898.31,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1421696 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1421696 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1421696 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 0.96,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1421696 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1421696 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 10653.18,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917572 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917572 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.34,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917572 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.67,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917572 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1917572 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2819.61,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=507529 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=507529 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=507529 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=507529 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=507529 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 12317.62,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2217172 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.26,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2217172 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2217172 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 2.99,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2217172 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2217172 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5246.21,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944317 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944317 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944317 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944317 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944317 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 7873.01,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1417141 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1417141 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1417141 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 0.97,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1417141 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1417141 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 10565.48,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1901787 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1901787 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.35,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1901787 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.7,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1901787 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1901787 Run: https://github.com/Phi-C/ATOM/actions/runs/29948654929"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0e9804c21ae050e4fc5627ff8c8d40390b63e2a5",
          "message": "review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error (#1549)\n\n* review-pr skill v2: dispatch gap rule, arch-constant FP fix, P5 timing error\n\nKey changes based on 191-PR test sweep (88 aiter + 93 ATOM):\n\n- B6 (new): \"New dispatch value not handled by all paths, no warning\"\n  Highest-frequency finding (~18% of PRs). Covers: new dtype/arch/flag/\n  getattr attribute added to a multi-way dispatch where some branches silently\n  fall through to wrong behavior with no assert/warning. FP self-check\n  included (upstream assert/isinstance guard → skip).\n  Real examples: ATOM#1548 (getattr silent zero), ATOM#841 (unmerged aiter kwarg).\n\n- C3 (new): \"New GPU arch string or arch-specific constant hardcoded\"\n  Extended beyond arch strings to cover magic constants tied to specific\n  model configs (e.g., hardcoded 576 for MLA kv_lora_rank+qk_rope_head_dim).\n  FP self-check front-loaded: search unchanged lines first; if constant\n  already exists → skip (pre-existing style).\n  Real example: ATOM#860 _bind_kv_cache_to_modules() hardcoded 576.\n\n- P5 (new): \"Benchmark timing excludes one-time setup cost\"\n  Covers: shuffle_weight, first-call JIT compile excluded from timing window\n  → claimed speedup is actually a regression when setup cost is included.\n\n- Step 3 classification: wired B6/P5 into trigger lines so new\n  constexpr/routing-flag PRs and perf PRs auto-prompt the right checks.\n  New aiter API usage PRs trigger B6 (new kwarg unhandled by all branches?).\n\n- Step 8 output format: findings now require three parts (Problem + Impact +\n  Action verb). \"Author must\" / \"Reviewer should ask\" required; no verb = do\n  not include in output.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr skill: B5→parametric family, G1b, D1b, C3 exemption, HK5 envs.py\n\n- B5 restructured into a 5-row parametric table (param-discard, param-removed,\n  attr-missing, dispatch-silent, rename) — aligned with aiter's B6 family;\n  ATOM-specific attr-missing sub-type added (getattr silent-zero fallback,\n  ATOM#1548 real example)\n- G1b: blocking queue.get() without timeout in production serving code;\n  ATOM#789 call_soon_threadsafe noted as correct pattern (not G1b)\n- D1b: Python-side UnboundLocalError from conditional variable assignment\n  (ATOM#860 real example)\n- C3: add capability guard exemption — arch string in dedicated _detect_*()\n  helper is not centralized dispatch hardcoding (ATOM#749)\n- HK5: updated to \"register in atom/envs.py AND document in README\"\n  (ATOM#749 finding: ATOM_NATIVE_TRITON_ATTN not in envs.py)\n- Step 3 classification: new trigger entries for G1b and D1b\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* review-pr: sync cross-file verification + CI-failure triage + verified/inferred discipline\n\nPorts the general (non-aiter-specific) v3 hardening from the aiter skill:\n- grep .cu+.cuh+.h, not just diff files, before claiming missing sync/branch\n- classify CI failures as infra/unrelated vs real (e.g. RTD docs failing on a .claude-only change is infra, ATOM#1549)\n- tag findings [verified]/[inferred]; never ship an unconfirmed root cause\n(aiter-only rules E4 downstream-CI / E5 owner-signoff intentionally NOT ported — ATOM is the downstream.)\n\n* review-pr: add global 🔴 gate — must name concrete triggering input before firing 🔴\n\nGeneralizes the per-rule FP self-check into a mandatory gate over all 🔴\nfindings, closing the gap on rules like D9 that omit one. A 🔴 with no\ndemonstrable triggering shape/scale/input must be downgraded or dropped.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix P5 — #4166 was a false positive, not a real example\n\naiter#4166 preshuffles the static weight once outside the timing loop and\nhonestly reports geomean 0.69x; it never claimed 1.14x. The old example\ntaught the amortization error. Reframe P5 to fire only on costs that recur\nper call / per cold start, and keep #4166 as a counter-example.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: Step 6 — add structural verification for AI-generated code\n\nThe description-smell table only pre-filters; AI fails structurally. Add six\naction-forcing checks that each yield a verified/inferred finding: hallucinated\n-symbol sweep, twin divergence, claim/comment↔code + number provenance, safety\ntheater, test-calibrated-to-pass, magic constant without derivation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: close Step-3 routing gaps + add FP self-check + staleness guard\n\n① Route D6/D7 (fake fns), D8 (contiguous), D9 (int32), D4 (invariant reversal)\n   into Step 3 — previously reachable only by scanning all of Section D.\n② Add pre-existing-search FP self-check to C1 (dtype hardcode), matching C3.\n③ Staleness guard on P2 production shapes.\n(ATOM has no tl.load-mask rule and no E4, so those aiter-side changes N/A here.)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: fix B5 (API-modification rule) — route it + widen public-API scope\n\nB5 (API propagation) had no Step-3 entry, and the \"aiter API change\" row pointed\nonly to B6 (new dispatch value), not B5. Route B5 via that row + a new \"API\nsignature change\" row. Fix param-removed/rename scope: a base-class or\nbridge-read signature change breaks all subclasses/bridges, not just same-file\n— widened and linked to E2 (plugin bridge sync).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* review-pr: add FP self-check to B5 rename/param-removed (found by live test on aiter#4227)\n\nMirror of the aiter B6 fix: a rename/removal behind a same-named wrapper / alias /\nre-export is backward-compatible and must not fire. Confirm no compatibility shim\nbefore firing.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: root <root@hjbog-srdc-24.amd.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-15T10:13:21Z",
          "url": "https://github.com/Phi-C/ATOM/commit/0e9804c21ae050e4fc5627ff8c8d40390b63e2a5"
        },
        "date": 1784836802493,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2072.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373124 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373124 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373124 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.61,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373124 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373124 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7406.1,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333098 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333098 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333098 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 5.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333098 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1333098 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3349.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=602850 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=602850 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=602850 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.91,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=602850 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=602850 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5149.74,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926954 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926954 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.24,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926954 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926954 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=926954 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6507.25,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171305 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171305 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171305 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171305 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1171305 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2090.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376374 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376374 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376374 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376374 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=376374 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7561.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361108 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361108 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.85,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361108 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361108 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1361108 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3448.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620780 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620780 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620780 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620780 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=620780 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5164.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929572 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929572 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929572 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929572 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=929572 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6502.71,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170487 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170487 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170487 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.79,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170487 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1170487 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2131.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383609 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383609 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383609 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383609 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=383609 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7478.97,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346214 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.09,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346214 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.91,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346214 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.97,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346214 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1346214 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3443.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619807 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619807 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619807 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619807 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=619807 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5170.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930729 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930729 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930729 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.55,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930729 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=930729 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6460.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162896 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.2,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162896 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162896 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.78,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162896 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1162896 Run: https://github.com/Phi-C/ATOM/actions/runs/30035865140"
          }
        ]
      }
    ]
  }
}