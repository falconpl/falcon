#
# The Falcon Programming Language
#
# dbus - Falcon - DBUS official binding
#
#   GNU Makefile for Linux, BSD and generic unices
#
#

#Define here the list of files to be compiled

OBJECTS= build/dbus.o \
			build/dbus_ext.o \
			build/dbus_st.o \
			build/dbus_mod.o \
			build/dbus_srv.o

all: builddir dbus.so

dbus.so: $(OBJECTS)
	g++ $(LDFLAGS) -o dbus.so $(OBJECTS) $$(falcon-conf -l)

build/%.o : src/%.cpp src/*.h
	g++ -c $$(falcon-conf -c) $(CXXFLAGS) $< -o $@

builddir:
	mkdir -p build

clean:
	rm -rf build
	rm -rf dbus.so


.PHONY: clean builddir
