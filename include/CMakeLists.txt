#
# Just install include files so that every part of the project can use it.
#

SET( fasm_files
   fasm/clexer.h
   fasm/comp.h
   fasm/labeldef.h
   fasm/pseudo.h
)

SET( falcon_files
falcon/allocator.h
falcon/autocstring.h
falcon/autowstring.h
falcon/basealloc.h
falcon/bufferedstream.h
falcon/carray.h
falcon/cclass.h
falcon/cdict.h
falcon/citerator.h
falcon/cobject.h
falcon/common.h
falcon/compiler.h
falcon/core_ext.h
falcon/deferrorhandler.h
falcon/deptab.h
falcon/destroyable.h
falcon/detmempool.h
falcon/dir_sys.h
falcon/dir_sys_unix.h
falcon/dir_sys_win.h
falcon/dll_base.h
falcon/dll_dl.h
falcon/dll.h
falcon/dll_mac.h
falcon/dll_win.h
falcon/enginedata.h
falcon/engine.h
falcon/engstrings.h
falcon/errhand.h
falcon/error.h
falcon/fassert.h
falcon/fbom.h
falcon/filestat.h
falcon/flcloader.h
falcon/format.h
falcon/fstream.h
falcon/fstream_sys_unix.h
falcon/fstream_sys_win.h
falcon/garbageable.h
falcon/gencode.h
falcon/generator.h
falcon/genericlist.h
falcon/genericmap.h
falcon/genericvector.h
falcon/genhasm.h
falcon/gentree.h
falcon/gpage.h
falcon/heap.h
falcon/heap_linux.h
falcon/heap_win.h
falcon/item.h
falcon/itemid.h
falcon/itempage.h
falcon/itemtraits.h
falcon/itm_deref.h
falcon/lineardict.h
falcon/linemap.h
falcon/llist.h
falcon/ltree.h
falcon/memblock.h
falcon/membuf.h
falcon/memory.h
falcon/mempool.h
falcon/messages.h
falcon/modloader.h
falcon/module.h
falcon/pagedict.h
falcon/pcodes.h
falcon/pointerpage.h
falcon/proptable.h
falcon/refcount.h
falcon/refstring.h
falcon/rosstream.h
falcon/runtime.h
falcon/service.h
falcon/setup.h
falcon/spage.h
falcon/src_lexer.h
falcon/stackframe.h
falcon/stdstreams.h
falcon/stream.h
falcon/string.h
falcon/stringstream.h
falcon/stringstream_srv.h
falcon/stringwithid.h
falcon/strtable.h
falcon/symbol.h
falcon/symlist.h
falcon/symtab.h
falcon/syntree.h
falcon/sys.h
falcon/testsuite.h
falcon/timestamp.h
falcon/time_sys.h
falcon/time_sys_unix.h
falcon/time_sys_win.h
falcon/traits.h
falcon/transcoding.h
falcon/types.h
falcon/uintpair.h
falcon/userdata.h
falcon/vmcontext.h
falcon/vm.h
falcon/vmmaps.h
falcon/xtree_fix.h
)

INSTALL( FILES ${fasm_files}
DESTINATION $ENV{FALCON_BUILD_TREE}/include/fasm)

INSTALL( FILES ${falcon_files}
DESTINATION $ENV{FALCON_BUILD_TREE}/include/falcon)

