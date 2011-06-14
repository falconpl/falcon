/*
   FALCON - The Falcon Programming Language.
   FILE: globalsvector.h

   Specialized vector holding global variables
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Feb 2011 11:10:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_GLOBALSVECTOR_H
#define FALCON_GLOBALSVECTOR_H

#include <falcon/setup.h>
#include <falcon/item.h>
#include <vector>

namespace Falcon
{

class GlobalSymbol;
class String;

/** Set of items with reference counting.
 */
class GlobalsVector
{
public:
   typedef std::vector<Item>::size_type size_type;

   GlobalsVector( size_type size = 0 );
   
   ~GlobalsVector();

   /** Resizes the vector. */
   void resize( size_type size );

   /** Increment the reference count of this object */
   void incref() const;

   /** Decrements the reference count of this object.
    The error must be considered invalid after this call.
    */
   void decref();

   /** Returns a new global symbol extracted from the required position.
   */
   GlobalSymbol* makeGlobal( const String& name, size_type pos );

private:
   typedef std::vector<Item> ItemVector;
   ItemVector m_vector;

   int32 m_refcount;
};

}

#endif