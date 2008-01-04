/*
   FALCON - The Falcon Programming Language.
   FILE: vmmaps.hm_currentModule

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
#include <falcon/module.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Symbol;


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


/** Instance of a live module entity.

   The VM sees modules as a closed, read-only entity. Mutable data in a module is actually
   held in a per-module map in the VM.

   This class helds a reference to a module known by the VM and to the variable data
   known in the module.

   A module can be entered only once in the VM, and it is uniquely identified by a name.
*/

class FALCON_DYN_CLASS LiveModule: public BaseAlloc
{
   Module *m_module;
   ItemVector m_globals;

public:
   LiveModule( Module *mod );
   ~LiveModule();

   const Module *module() const { return m_module; }
   const ItemVector &globals() const { return m_globals; }
   ItemVector &globals() { return m_globals; }

   /** Just a shortcut to the name of the module held by this LiveModule. */
   const String &name() const { return m_module->name(); }

   /** Just a shortcut to the source of the module held by this LiveModule. */
   const byte *code() const { return m_module->code(); }

};


class LiveModulePtrTraits: public ElementTraits
{
public:
	virtual uint32 memSize() const;
	virtual void init( void *itemZone ) const;
	virtual void copy( void *targetZone, const void *sourceZone ) const;
	virtual int compare( const void *first, const void *second ) const;
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

/** Map of active modules in this VM.
   (const String *, LiveModule * )
*/
class FALCON_DYN_CLASS LiveModuleMap: public Map
{
public:
   LiveModuleMap();
};


/** Pair of the symbol and the module it is declared in.
   This is just a commodity class used to store the association between a certain symbol and the module
   it came from given the VM viewpoint (that is, the ID of the source module in the VM module list.
*/

class FALCON_DYN_CLASS SymModule: public BaseAlloc
{
   Symbol *m_symbol;
   LiveModule *m_lmod;
   Item *m_item;

public:
   SymModule( Item *itm, LiveModule *mod, Symbol *sym ):
      m_item( itm ),
      m_symbol( sym ),
      m_lmod( mod )
   {}

   Item *item() const { return m_item; }
   Symbol *symbol() const { return m_symbol; }
   uint32 symbolId() const { return m_symbol->itemId(); }
   LiveModule *liveModule() const { return m_lmod; }
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


namespace traits
{
   extern SymModuleTraits t_SymModule;
}

/** Map of symbol names and module where they are located.
   (const String *, SymModule )
*/
class FALCON_DYN_CLASS SymModuleMap: public Map
{
public:
   SymModuleMap();
};

}

#endif

/* end of vmmaps.h */
