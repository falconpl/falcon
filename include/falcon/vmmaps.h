/*
   FALCON - The Falcon Programming Language.
   FILE: flc_vmmaps.h
   $Id: vmmaps.h,v 1.7 2007/08/05 22:49:46 jonnymind Exp $

   Map items used in VM and related stuff
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer ott 20 2004
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
   Map items used in VM and related stuff.
*/

#ifndef flc_vmmaps_H
#define flc_vmmaps_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/genericvector.h>
#include <falcon/genericmap.h>
#include <falcon/itemtraits.h>
#include <falcon/symbol.h>
#include <falcon/string.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Symbol;
class Module;

/** Pair of the symbol and the module it is declared in.
   This is just a commodity class used to store the association between a certain symbol and the module
   it came from given the VM viewpoint (that is, the ID of the source module in the VM module list.
*/

class SymModule: public BaseAlloc
{
   Symbol *m_symbol;
   uint32 m_moduleId;
   Item *m_item;
public:
   SymModule( Item *itm, uint32 mod, Symbol *sym ):
      m_item( itm ),
      m_symbol( sym ),
      m_moduleId( mod )
   {}

   Item *item() const { return m_item; }
   Symbol *symbol() const { return m_symbol; }
   uint32 symbolId() const { return m_symbol->itemId(); }
   uint32 moduleId() const { return m_moduleId; }
};

class SymModuleTraits: public ElementTraits
{
public:
	virtual uint32 memSize() const;
	virtual void init( void *itemZone ) const;
	virtual void copy( void *targetZone, const void *sourceZone ) const;
	virtual int compare( const void *first, const void *second ) const;
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

/** Map of symbol names and module where they are located.
   (const String *, SymModule )
*/
class FALCON_DYN_CLASS SymModuleMap: public Map
{
public:
   SymModuleMap();
};

class FALCON_DYN_CLASS ItemVector: public GenericVector
{
   const ItemTraits s_it;

public:
   ItemVector( uint32 prealloc=0 ):
      GenericVector()
  {
      init( &s_it, prealloc );
  }

   Item &itemAt( uint32 pos ) const { return *(Item *) at( pos ); }
   Item *itemPtrAt( uint32 pos ) const { return (Item *) at( pos ); }
   void setItem( const Item &item, uint32 pos ) { set( const_cast<Item *>(&item), pos ); }
   Item &topItem() const { return *(Item *) top(); }
};

class FALCON_DYN_CLASS GlobalsVector: public GenericVector
{
   const VoidpTraits s_vpt;
public:
   GlobalsVector( uint32 prealloc=0 ):
      GenericVector()
   {
      init( &s_vpt, prealloc );
   }

   ItemVector &vat( uint32 pos ) const { return **(ItemVector **) at( pos ); }
   ItemVector *vpat( uint32 pos ) const { return *(ItemVector **) at( pos ); }
   ItemVector &topvp() const { return **(ItemVector **) top(); }
};


namespace traits
{
   extern SymModuleTraits t_SymModule;
}

}

#endif

/* end of flc_vmmaps.h */
