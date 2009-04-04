/*
   FALCON - The Falcon Programming Language.
   FILE: vmmaps.hm_currentModule

   The Representation of module live data once linked in a VM
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 03 Apr 2009 23:09:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The Representation of module live (dynamic) data once linked in a VM.
*/

#ifndef FALCON_LIVEMODULE_H
#define FALCON_LIVEMODULE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/genericvector.h>
#include <falcon/genericmap.h>
#include <falcon/itemtraits.h>
#include <falcon/symbol.h>
#include <falcon/string.h>
#include <falcon/module.h>
#include <falcon/basealloc.h>
#include <falcon/vmmaps.h>

namespace Falcon
{

/** Instance of a live module entity.

   The VM sees modules as a closed, read-only entity. Mutable data in a module is actually
   held in a per-module map in the VM.

   This class helds a reference to a module known by the VM and to the variable data
   known in the module.

   A module can be entered only once in the VM, and it is uniquely identified by a name.

   This class acts also as a weak reference between live callable items and modules.
   When a live module is unlinked, the contents of this class are zeroed and
   every callable item referencing this module becomes a nil as isCallable()
   gets called.

   This object is garbageable; it gets referenced when it's in the module map and by
   items holding a callable in this module. When a module is unlinked, the LiveModule
   may be destroyed when it is not referenced in garbage anymore, or at VM termination.

   \note Although this is a Garbageable item, and as such, it resides in the memory pool,
   there isn't any need for precise accounting of related memory, as globals are allocated
   and destroyed when the module is linked or unlinked, and not (necessarily) when this
   item is collected. As the memory associated with this object is separately reclaimed
   by detachModule(), and it cannot possibly be reclaimed during a collection loop,
   there is no meaning in accounting it.
*/

class FALCON_DYN_CLASS LiveModule: public Garbageable
{
   Module *m_module;
   ItemVector m_globals;
   ItemVector m_wkitems;
   bool m_bPrivate;

   String** m_strings;
   uint32 m_stringCount;

public:
   typedef enum {
      init_none,
      init_trav,
      init_complete
   } t_initState;

private:
   t_initState m_initState;

public:
   LiveModule( Module *mod, bool bPrivate=false );
   ~LiveModule();

   virtual bool finalize();

   const Module *module() const { return m_module; }
   const ItemVector &globals() const { return m_globals; }
   ItemVector &globals() { return m_globals; }

   const ItemVector &wkitems() const { return m_wkitems; }
   ItemVector &wkitems() { return m_wkitems; }

   /** Just a shortcut to the name of the module held by this LiveModule. */
   const String &name() const { return m_module->name(); }

   /** Disengage a module after a module unlink. */
   void detachModule();

   /** Is this module still alive?
       Short for this->module() != 0
       \return true if the module is still alive.
   */
   bool isAlive() const { return m_module != 0; }

   /** Return a module item given a global symbol name.
      This is an utility funtion retreiving an global item declared by the
      module that is referenced in this live data.
      \param symName the name of the global symbol for which this item must be found
      \return 0 if not found or a pointer to the item which is indicated by the symbol
   */
   Item *findModuleItem( const String &symName ) const;

   /** Returns the privacy status of this module.

      If the module is not private, the VM will export all the exported symbol
      to the global namespace at link time.

      \return true if this module is private, and so, if didn't export any symbol.
   */
   bool isPrivate() const { return m_bPrivate; }

   /** Changes the privacy status of this module.

      If changing private to public, the VM funcrion exporting public symbols
      must be separately called.
   */
   void setPrivate( bool mode ) { m_bPrivate = mode; }

   t_initState initialized() const { return m_initState; }
   void initialized( t_initState tis ) { m_initState = tis; }

   /** Return the string in the module with the given ID.
   */
   String* getString( uint32 stringId ) const;
};


class LiveModulePtrTraits: public ElementTraits
{
public:
   virtual ~LiveModulePtrTraits() {}
	virtual uint32 memSize() const;
	virtual void init( void *itemZone ) const;
	virtual void copy( void *targetZone, const void *sourceZone ) const;
	virtual int compare( const void *first, const void *second ) const;
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

}

#endif

/* end of livemodule.h */
