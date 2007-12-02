/*
   FALCON - The Falcon Programming Language.
   FILE: flc_cdict.h
   $Id: cdict.h,v 1.8 2007/08/11 00:11:51 jonnymind Exp $

   Core dictionary -- base abstract class for dictionary interfaces.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab dic 4 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Core dictionary -- base abstract class for dictionary interfaces.
*/

#ifndef flc_cdict_H
#define flc_cdict_H

#include <falcon/types.h>
#include <falcon/citerator.h>
#include <falcon/garbageable.h>
#include <falcon/item.h>

namespace Falcon {

class FALCON_DYN_CLASS DictIterator: public CoreIterator
{
protected:
   DictIterator() {}

public:
   virtual const Item &getCurrentKey() const = 0;
};



class FALCON_DYN_CLASS CoreDict: public Garbageable
{

protected:
   CoreDict( VMachine *vm ):
      Garbageable(vm)
   {}

   CoreDict( VMachine *vm, uint32 alloc ):
      Garbageable( vm, alloc )
   {}

public:
   //=======================================
   // Public overridable interface
   //

   virtual uint32 length() const =0;
   virtual Item *find( const Item &key ) = 0;
   virtual bool find( const Item &key, DictIterator &iter ) = 0;
   virtual DictIterator *findIterator( const Item &key ) = 0;

   virtual bool remove( DictIterator &iter ) = 0;
   virtual bool remove( const Item &key ) = 0;
   virtual void insert( const Item &key, const Item &value ) = 0;
   virtual void smartInsert( DictIterator &iter, const Item &key, const Item &value ) = 0;

   virtual void first( DictIterator &iter ) = 0;
   virtual void last( DictIterator &iter ) = 0;
   virtual DictIterator *first() = 0;
   virtual DictIterator *last() = 0;

   virtual bool equal( const CoreDict &other ) const = 0;
   virtual CoreDict *clone() const = 0;
   virtual void merge( const CoreDict &dict ) = 0;
   virtual void clear() = 0;


   /** Generic traversal interface.
      Usually, dictionary traversal is needed by VM or other engine related classes.
   */
   virtual void traverseBegin() = 0;
   virtual bool traverseNext( Item &key, Item &value ) = 0;

   //=======================================
   // Utilities

   virtual bool find( const Item &key, Item &value )
   {
      Item *itm;
      if( ( itm = find( key ) ) != 0 )
      {
         value = *itm;
         return true;
      }
      return false;
   }

   bool empty() const { return length() == 0; }

};

}

#endif

/* end of flc_cdict.h */
