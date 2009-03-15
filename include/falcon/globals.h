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
   
   FALCON_DYN_SYM bool addVFS( const String &name, VFSProvider *prv );
   FALCON_DYN_SYM VFSProvider* getVFS( const String &name );
   
   const String &getMessage( uint32 id );
   bool setTable( StringTable *tab );
   bool setLanguage( const String &language );

   class AutoInit {
   public:
      inline AutoInit() { Init(); }
      inline ~AutoInit() { Shutdown(); }
   };
   
}

}
#endif

/* end of globals.h */
