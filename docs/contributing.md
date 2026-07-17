# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Development setup

This repository uses [pixi](https://pixi.sh) for environment and task management:

```console
git clone https://github.com/serapeum-org/geostatista.git
cd geostatista
pixi install -e dev
pixi run -e dev main        # run the test suite
pixi run -e docs docs-serve # serve the docs locally
```

Install the pre-commit hooks once so formatting and linting run on every commit:

```console
pixi run -e dev pre-commit install
```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build.
2. Update the docs with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
3. Versioning is handled by [Commitizen](https://commitizen-tools.github.io/commitizen/) from the
   conventional-commit history; do not bump the version by hand. The versioning scheme we use is
   [SemVer](http://semver.org/) (staying in `0.x` until the API stabilizes).
4. You may merge the Pull Request in once you have the sign-off of another developer, or if you do
   not have permission to do that, you may request a reviewer to merge it for you.

## Code of Conduct

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) that governs all interactions in this
project.
