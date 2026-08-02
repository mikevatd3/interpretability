# interp-storage

Shared DigitalOcean Spaces (S3-compatible) read/write helpers, used across
every experiment in this repo rather than copy-pasted per experiment --
see the root `README.md`'s "Shared infrastructure" section for why this
one thing is shared while most experiment code deliberately isn't.

## Use from an experiment

```
cd <your experiment folder>
uv add --editable ../interp_storage
```

Then in code:
```python
from interp_storage import sync_down, sync_up

sync_down(some_local_path)   # pull from Spaces if configured, else no-op
sync_up(some_local_path)     # push to Spaces if configured, else no-op
```

And from the command line, to seed the bucket with a file that doesn't
exist there yet (fails loudly if Spaces env vars aren't set, unlike
`sync_up` itself):
```
uv run load-to-spaces <path> [<path> ...]
```

See the root `README.md`'s "DigitalOcean Spaces" section for the required
`.env` keys.
