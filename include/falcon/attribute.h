/*
   FALCON - The Falcon Programming Language.
   FILE: attribute.h

   Standard VM attriubte item.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef FLC_ATTRIBUTE_H
#define FLC_ATTRIBUTE_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/string.h>
#include <falcon/symbol.h>
#include <falcon/citerator.h>
#include <falcon/item.h>


namespace Falcon
{

// forward decl
class AttribHandler;
class CoreObject;
class AttribIterator;

/** Class holding pointers to objects.
   Internally used by attributes to store lists of objects.

   As the object destructor grants that any object handler stored in attributes
   pointing to them is destroyed, there is no need for reference counting.
*/
class FALCON_DYN_CLASS AttribObjectHandler: public BaseAlloc
{
   CoreObject *m_object;
   AttribObjectHandler *m_next;
   AttribObjectHandler *m_prev;

public:
   AttribObjectHandler( CoreObject *obj, AttribObjectHandler *p = 0, AttribObjectHandler *n = 0 ):
      m_object( obj ),
      m_next( n ),
      m_prev( p )
   {}

   CoreObject *object() const { return m_object; }

   void next( AttribObjectHandler *n ) { m_next = n; }
   AttribObjectHandler *next() const { return m_next; }

   void prev( AttribObjectHandler *p ) { m_prev = p; }
   AttribObjectHandler *prev() const { return m_prev; }
};

/** Definition of Falcon attribute.
   An attribute is a binary object that can be "given" or "taken from" a live object.
   They are defined in the VM and are assigned to objects through GIVE requests.

   When an object is given an attribute, it also enters the list of attribute objects;
   from there, it is iterable for "object loops".

   Being in an attribute list doesn't prevent an object to be garbaged. At object destruction,
   the object is removed from the list of existing attributes.

   Attributes are also serialized and restored at de-serialization.

   An attribute can be given to an object or removed from an object; to test if an object
   has a certain attribute, it is either possible to search for the object in the list of
   stored object or to test for the attribute handler pointing to this attribute.

   Attributes keeps also track of the iterators, which can be created on a given attribute
   to allow iteration on the objects being categorized by this attribute.
*/
class FALCON_DYN_CLASS Attribute: public BaseAlloc
{
   const Symbol *m_symbol;
   AttribObjectHandler *m_head;
   AttribIterator *m_iterHead;
   uint32 m_size;

   void removeObject( AttribObjectHandler *hobj );

   friend class AttribIterator;

public:
   /** Creates the attribute instance.
      The attribute needs the symbol it is created for.
      Every attribute lives as a symbol in a module, so the VM can be sure there
      are no dupicate attributes, and that attribute assigments respect the
      import/export rules.
   */
   Attribute( const Symbol *sym );
   ~Attribute();

   /** Gets the name of the attribute.
      Shortcut to symbol name.
      \return the name of the symbol associated with this attribute.
   */
   const String &name() const { return m_symbol->name(); }

   const Symbol *symbol() const { return m_symbol; }

   /** Give this attribute to an object.
      \param tgt the object to which this attribute must be given.
      \return false if the attribute was already given to the object, false otherwise.
   */
   bool giveTo( CoreObject *tgt );

   /** Remove this attribute from an object.
      \param tgt the object from which this attribute must be remove.
      \return false if the attribute wasn't given to the object, false otherwise.
   */
   bool removeFrom( CoreObject *tgt );

   /** Check if this attribute has some object associated.
      \return true if there is some item associated, false otherwise.
   */
   bool empty() const { return m_head == 0; }

   /** Gets the head of the list of stored objects.
      \return true if there is some item associated, false otherwise.
   */
   AttribObjectHandler *head() const { return m_head; }

   /** Creates an iterator for this attribute.
      \return a newly allocated iterator which iterates on the objects being categorized
              on the categorized objects.
   */
   AttribIterator *getIterator();

   /** Return current count of objects having this attribute.
      \return count of objects being held.
   */
   uint32 size() const { return m_size; }

   /** Remove from all objects.
      This method removes the attribute from all objects in the owning VM.
   */
   void removeFromAll();

};


/** Class holding pointers to attributes.
   Internally used by objects to store the list of attributes they are assigned to.
*/
class FALCON_DYN_CLASS AttribHandler: public BaseAlloc
{
   Attribute *m_attrib;
   AttribObjectHandler *m_hobj;
   AttribHandler *m_next;
   AttribHandler *m_prev;

public:
   /** Initializes an attribute handler.
      The attribute handler has a mandatory pointer to the attribute it is a handler for,
      and may have a pointer to an object handler in the owner attribute. This makes links
      and unlinks much faster.

      The parameter is mandatory for attribute handlers being held in objects, but it is optional
      for attribute handlers used as list of attributes i.e. in the VM or in class definitions.
   */
   AttribHandler( Attribute *obj, AttribObjectHandler *mo, AttribHandler *p = 0, AttribHandler *n = 0 ):
      m_attrib( obj ),
      m_next( n ),
      m_prev( p ),
      m_hobj( mo )
   {}

   Attribute *attrib() const { return m_attrib; }
   AttribObjectHandler *objHandler() const { return m_hobj; }

   void next( AttribHandler *n ) { m_next = n; }
   AttribHandler *next() const { return m_next; }

   void prev( AttribHandler *p ) { m_prev = p; }
   AttribHandler *prev() const { return m_prev; }
};


class FALCON_DYN_CLASS AttribIterator: public CoreIterator
{
   Attribute *m_attrib;
   AttribObjectHandler *m_current;

   AttribIterator *m_next;
   AttribIterator *m_prev;

   mutable Item m_item;

public:

   AttribIterator( Attribute *attrib );
   ~AttribIterator();

   virtual bool next();
   virtual bool prev();
   virtual bool hasNext() const;
   virtual bool hasPrev() const;

   virtual Item &getCurrent() const;

   virtual bool isValid() const;
   virtual bool isOwner( void *collection ) const;
   virtual bool equal( const CoreIterator &other ) const;
   virtual void invalidate();
   virtual bool erase();
   virtual bool insert( const Item &item );
   virtual UserData *clone();

   AttribIterator *nextIter() const { return m_next; }
   void notifyDeletion( AttribObjectHandler *deleted );
};

}

#endif

/* end of attribute.h */
