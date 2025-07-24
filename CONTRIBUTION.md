# Contribution

## Support new provider

Providers for code understanding and generation is set up in `/config/code_understanding.json` and `generation.json`. Provider names in these setting files must be under the same naming convention. Otherwise, `server/config.py` may raise an error when reading from configuration files.