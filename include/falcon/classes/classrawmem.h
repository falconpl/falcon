/*
   FALCON - The Falcon Programming Language.
   FILE: classrawmem.h

   Handler for unformatted raw memory stored in the GC.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Jun 2012 16:52:30 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSRAWMEM_H
#define FALCON_CLASSRAWMEM_H

#include <falcon/class.h>

namespace Falcon {

/** Handler for unformatted raw memory stored in the GC. 
 This is just a class used to handle third party, unformatted raw memory
 stored in the garbage collector.
 
 The only requirement is that the assigned memory starts with an uint32
 sized free space that is used by this class to perform GC accounting.
 
 If the final program can use use arbitrary memory that can be created by
 the Falcon engine, this class can be used directly. In case this is not possible, 
 use the ClassData extension, which provides a way to point to arbitrary memory 
 that is allocated outside the scope of Falcon.
 
 */
class FALCON_DYN_CLASS ClassRawMem: public Class
{
public:
   ClassRawMem();
   virtual ~ClassRawMem();
   
   virtual void* createInstance() const; 
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   void* allocate( uint32 size ) const;
};

}

#endif	/* CLASSRAWMEM_H */

