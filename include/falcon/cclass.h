/*
   FALCON - The Falcon Programming Language.
   FILE: cclass.h

   Core Class definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu Jan 20 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
class AttribHandler;
class Attribute;

/** Representation of classes held in VM while executing code.
   The Virtual Machine has his own image of the classes. Classes items
   are meant to i.e. access classes wide variables or classes methods.

   Class Items are also used to create objects in a faster way. They
   maintain a copy of an empty object, that is just duplicated as-is
   via memcopy, making the creation of a new object a quite fast operation.

   They also store a list of attributes that must be given at object
   after their creation. As the classes doesn't really has attributes,
   they do not participate in attribute loops and list; they just have
   to remember which attributes must be given to objects being constructed
   out of them.

*/

class FALCON_DYN_CLASS CoreClass: public Garbageable
{

private:
   LiveModule *m_lmod;
   const Symbol *m_sym;
   Item m_constructor;
   PropertyTable *m_properties;
   AttribHandler *m_attributes;

   /** Cached object manager */
   ObjectManager *m_manager;

   /** Configures a newborn instance.
      This method is called by createInstance on this class and all the subclasses,
      provided they have non-default object managers.

      If a class has the default object manager (0), then it shares the common property
      array that makes up the object structure through the PropertyManager class, and no
      extra configuration is needed.

      This method may call reflected classes constructor to configure their user data, but
      it won't call the "init" sequence of the object. That is left to the VM or to the
      user, and should be done after the object is configured.
   */
   void configInstance( CoreObject *co );

public:

   /** Creates an item representation of a live class.
      The representation of the class nees a bit of extra informations that are provided by the
      virtual machine, other than the symbol that generated this item.
      The module id is useful as this object often refers to its module in the VM. Having the ID
      recorded here prevents the need to search for the live ID in the VM at critical times.
   */
   CoreClass( VMachine *mp, const Symbol *sym, LiveModule *lmod, PropertyTable *pt );
   ~CoreClass();

   LiveModule *liveModule() const { return m_lmod; }
   const Symbol *symbol() const { return m_sym; }

   PropertyTable &properties() { return *m_properties; }
   const PropertyTable &properties() const { return *m_properties; }


   /** Creates an instance of this class.
      The returned object and all its properties are stored in the same memory pool
      from which this instance is created.
      On a multithreading application, this method can only be called from inside the
      thread that is running the VM.

      The instance may be created with some pre-configured user-data (which must be
      manageable by the ObjectManager declared by this class); if a user data is not
      provided, or if it's zero, the class will create a user data using it's own
      ObjectManager.

      In some cases (e.g. de-serialization) the caller may wish not to have
      the object initialized and filled with attributes. The optional
      appedAtribs parameter may be passed false to have the caller to
      fill the instance with startup data.

      \note This function never calls the constructor of the object.
      \param user_data pre-allocated user data.
      \param appendAttribs false to prevent initialization of default attributes.
      \return The new instance of the Core Object.
   */
   CoreObject *createInstance( void *user_data=0, bool appendAttribs = true ) const;

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

   /** Adds an attribute to a class definition.
      This attribute will be given to every instance of this class.

      There is no validity check about i.e. attribute duplication as
      this function is usually called by the VM at link time on a properly
      compiled attribute list, or by extensions at module creation time.

      Also, duplicate attributes would have no consequence.
      \param attrib the attribute do be added to this class definition.
   */
   void addAttribute( Attribute *attrib );

   /** Removes an attribute to a class definition.
      Removes a previously given attribute.

      Generally used by the VM at class inheritance resolution time.
      \param attrib the attribute to be removed.
   */
   void removeAttribute( Attribute *attrib );

   /** Changes the attribute list.
      This is mainly used by the VM during the link phase.
   */
   void setAttributeList( AttribHandler *lst );

   /** Returns the object manager used by this class.
      Shortcut that gets the manager declared by the topmost reflective class.
      \return the ObjectManager used for this class.
   */
   ObjectManager* getObjectManager() const { return m_manager; }

   /** Sets  the object manager used by this class.
      Used directly by the VM during the link phase.
   */
   void setObjectManager( ObjectManager *om ) { m_manager = om; }
};

}

#endif

/* end of flc_cclass.h */
