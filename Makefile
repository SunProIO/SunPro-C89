REVIEWS = $(shell find . -name '*.src-re' | sed 's/ /\\ /g')
SOURCES = $(REVIEWS:.src-re=.re)
XMLS = $(REVIEWS:.src-re=.xml)
HTMLS = $(REVIEWS:.src-re=.html)

SRCDIR = src

NPMDIR = $(shell npm bin)

BE = bundle exec

build: clean all
	git submodule init
	git submodule update

clean:
	rm $(SOURCES) $(XMLS) $(HTMLS) -f

%.re: %.src-re
	$(NPMDIR)/lsc scripts/build.ls "$<" "$@"

%.rawxml: %.re
	$(BE) review-compile --target idgxml "$<" --yaml config.yml > "$@"

%.xml: %.rawxml
	sed -e 's/file:\/\/src/file:\/\/\/C:\/Users\/hakatashi\/Documents\/src/g' "$<" > "$@"

%.html: %.src-re
	cp "$<" "$(basename $<).re"
	cp --parents layouts/layout.html.erb "$(dir $<)"
	cd "$(dir $<)"; $(BE) review-compile --target html "$(notdir $<)" --yaml ../../config.yml > "$(notdir $@)"

html: $(HTMLS)

xml: $(XMLS)

all: $(HTMLS) $(XMLS)
