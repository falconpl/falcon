/*
   FALCON - The Falcon Programming Language.
   FILE: livemodule.h

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
#include <falcon/carray.h>
#include <falcon/genericmap.h>
#include <falcon/symbol.h>
#include <falcon/string.h>
#include <falcon/module.h>
#include <falcon/basealloc.h>
#include <falcon/strtable.h>

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
   
   The live module keeps also track of strings taken from the underlying module
   string table and injected in the live VM. In this way, the strings become
   independent from the underlying module, that can be unloaded while still
   sharing string data with the host VM.

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
   mutable CoreString** m_strings;
   mutable uint32 m_strCount;
   mutable uint32 m_aacc;
   mutable int32 m_iacc;
   ItemArray m_globals;
   ItemArray m_wkitems;
   bool m_bPrivate;
   bool m_bAlive;
   bool m_needsCompleteLink;

   ItemArray m_userItems;

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
   const ItemArray &globals() const { return m_globals; }
   ItemArray &globals() { return m_globals; }

   const ItemArray &wkitems() const { return m_wkitems; }
   ItemArray &wkitems() { return m_wkitems; }

   /** Just a shortcut to the name of the module held by this LiveModule. */
   const String &name() const { return m_module->name(); }

   /** Disengage a module after a module unlink. */
   void detachModule();

   /** Is this module still alive?
       Short for this->module() != 0
       \return true if the module is still alive.
   */
   bool isAlive() const { return m_bAlive; }

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
   
   /** True if this module requires a second link step. */
   bool needsCompleteLink() const { return m_needsCompleteLink; }
   
   /** True if this module requires a second link step. */
   void needsCompleteLink( bool l ) { m_needsCompleteLink = l; }
   
   void gcMark( uint32 mark );

   /** Returns the user items of this live modules.
    *
    * Modules often need a module-global storage. However, the virtual
    * machine doesn't offer this space, and using global variables requires
    * a bit of gym to get the variable ID and the item in the live
    * module global array.
    *
    * The module functions, at extension level (i.e. the C++ implementations)
    * can take advantage of this item array which is dedicated to them.
    *
    * The array is also autonomously GC marked.
    */
   ItemArray& userItems() { return m_userItems; }
   const ItemArray& userItems() const { return m_userItems; }
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
