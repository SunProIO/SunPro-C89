REVIEWS = $(shell find . -name '*.re' | sed 's/ /\\ /g')
SOURCES = $(REVIEWS:.re=.build-re)
XMLS = $(REVIEWS:.re=.xml)

SRCDIR = src

NPMDIR = $(shell npm bin)

BE = bundle exec

build: clean all
	git submodule init
	git submodule update

clean:
	rm $(SOURCES) -rf
	rm $(XMLS) -rf

%.build-re: %.re
	$(NPMDIR)/lsc scripts/build.ls "$<" "$@"

%.xml: %.build-re
	$(BE) review-compile --target idgxml "$<" > "$@"

all: $(XMLS)
