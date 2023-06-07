compile:
	rm -rf ./dist/ && cp -r ./docs/ ./dist/ && cp -r ./src/ ./dist/ && cp -r ./dist/_shared/docs/ ./dist/ && rm -rf ./dist/_build/ && poetry run nbdoc_build --srcdir ./dist/

docs:
	yarn start
