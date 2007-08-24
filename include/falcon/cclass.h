/*
   FALCON - The Falcon Programming Language.
   FILE: flc_cclass.h
   $Id: cclass.h,v 1.6 2007/08/11 00:11:51 jonnymind Exp $

   Core Class definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu Jan 20 2005
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
   Core Class definition
*/

#ifndef flc_cclass_H
#define flc_cclass_H

#include <falcon/setup.h>
#include <falcon/symbol.h>
#include <falcon/proptable.h>
#include <falcon/mempool.h>
#include <stdlib.h>

namespace Falcon {

class VMachine;

/** Representation of classes held in VM while executing code.
   The Virtual Machine has his own image of the classes. Classes items
   are meant to i.e. access classes wide variables or classes methods.

   Class Items are also used to create objects in a faster way. They
   maintain a copy of an empty object, that is just duplicated as-is
   via memcopy, making the creation of a new object a quite fast operation.

*/

class FALCON_DYN_CLASS CoreClass: public Garbageable
{

private:
   uint32 m_modId;
   Symbol *m_sym;
   Item m_constructor;
   PropertyTable *m_properties;
   uint64 m_attributes;

public:

   /** Creates an item representation of a live class.
      The representation of the class nees a bit of extra informations that are provided by the
      virtual machine, other than the symbol that generated this item.
      The module id is useful as this object often refers to its module in the VM. Having the ID
      recorded here prevents the need to search for the live ID in the VM at critical times.

   */
   CoreClass( VMachine *mp, Symbol *sym, int modId, PropertyTable *pt );
   ~CoreClass();

   uint32 moduleId() const { return m_modId; }
   Symbol *symbol() const { return m_sym; }

   PropertyTable &properties() { return *m_properties; }
   const PropertyTable &properties() const { return *m_properties; }


   /** Creates an instance of this class.
      The returned object and all its properties are stored in the same memory pool
      from which this instance is created.
      On a multithreading application, this method can only be called from inside the
      thread that is running the VM.
   */
   CoreObject *createInstance() const;

   /** Special constructor for uncollectable instances.
       This method is meant to be called when the original VM cann't be manipulated,
       i.e. from foreign threads.
       The returned object must be either disposed with Item::destroy() (after
       having entered an item) or stored with storeForGarbageDeep() on the target VM.
   */
   CoreObject *createUncollectedInstance() const;

   uint64 attributes() const { return m_attributes; }
   void attributes( uint64 data ) { m_attributes = data; }

   const Item &constructor() const { return m_constructor; }
   Item &constructor() { return m_constructor; }

   /** Returns true if the class is derived from a class with the given name.
      This function scans the property table of the class (template properties)
      for an item with the given name, and if that item exists and it's a class
      item, then this method returns true.
      \param className the name of a possibly parent class
      \return true if the class is derived from a class having the given name
   */
   bool derivedFrom( const String &className ) const;
};

}

#endif

/* end of flc_cclass.h */
