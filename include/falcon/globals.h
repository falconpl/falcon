/*
   FALCON - The Falcon Programming Language.
   FILE: globals.h

   Engine-wide exported variables and global functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 02 Mar 2009 20:19:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef FALCON_GLOBALS_H
#define FALCON_GLOBALS_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <stdlib.h>  // for size_t

namespace Falcon
{

class MemPool;
class StringTable;
class VFSProvider;
class String;

FALCON_DYN_SYM extern void * (*memAlloc) ( size_t );
FALCON_DYN_SYM extern void (*memFree) ( void * );
FALCON_DYN_SYM extern void * (*memRealloc) ( void *,  size_t );

FALCON_DYN_SYM extern void * (*gcAlloc) ( size_t );
FALCON_DYN_SYM extern void (*gcFree) ( void * );
FALCON_DYN_SYM extern void * (*gcRealloc) ( void *,  size_t );
FALCON_DYN_SYM extern MemPool *memPool;
FALCON_DYN_SYM extern StringTable *engineStrings;

class MemPool;

namespace Engine
{
   FALCON_DYN_SYM void Init();
   FALCON_DYN_SYM void Shutdown();

   /** Utility function recording the preferential encodings for sources and VM I/O.
      When the engine has to create its own VMs and streams, i.e. to fulfil
      interactive compiler requests, it uses this encodings that can
      be defined and changed by the application at any time.

      The values can be used also by the calling application as a convenient
      inter-moule communication area for this critical aspect of preferential
      encoding.

      \param sSrcEnc The encoding preferentially used by source files.
      \param sIOEnc The encoding preferentially used in I/O streams different from sources.
   */
   FALCON_DYN_SYM void setEncodings( const String &sSrcEnc, const String &sIOEnc );

   /** Utility function recording the preferential encodings for sources and VM I/O.
      \see recordEncodings

      \param sSrcEnc The encoding preferentially used by source files.
      \param sIOEnc The encoding preferentially used in I/O streams different from sources.
   */
   FALCON_DYN_SYM void getEncodings( String &sSrcEnc, String &sIOEnc );

   FALCON_DYN_SYM bool addVFS( const String &name, VFSProvider *prv );
   FALCON_DYN_SYM VFSProvider* getVFS( const String &name );

   FALCON_DYN_SYM const String &getMessage( uint32 id );
   FALCON_DYN_SYM bool setTable( StringTable *tab );
   FALCON_DYN_SYM bool setLanguage( const String &language );
   /** Set application wide search path.
      This is used by default in new VMs, module loaders and metacompilers.
   */
   FALCON_DYN_SYM void setSearchPath( const String &path );
   
   /** Returns the application-wide default search path by copy. */
   FALCON_DYN_SYM String getSearchPath();

   class AutoInit {
   public:
      inline AutoInit() { Init(); }
      inline ~AutoInit() { Shutdown(); }
   };

}

}
#endif

/* end of globals.h */
