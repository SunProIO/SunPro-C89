REVIEWS = $(shell find . -name '*.re' | sed 's/ /\\ /g')
SOURCES = $(REVIEWS:.re=.build.re)
XMLS = $(REVIEWS:.re=.xml)

SRCDIR = src

BE = bundle exec

build: clean all
	git submodule init
	git submodule update
	cp $(SRCDIR) $(BUILDDIR) -r

clean:
	rm $(SOURCES) -rf
	rm $(XMLS) -rf

%.build-re: %.re
	cp "$<" "$@"

%.xml: %.build-re
	$(BE) review-compile --target idgxml "$<" > "$@"

all: $(XMLS)
