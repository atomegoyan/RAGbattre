#!/bin/bash
# entrypoint.sh

# Take ownership of the Hugging Face cache directory
chown -R appuser:appuser /home/appuser/.cache/huggingface

# Execute the original command as appuser
exec gosu appuser "$@"
