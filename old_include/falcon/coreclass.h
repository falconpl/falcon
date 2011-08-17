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

#ifndef FLC_CORECLASS_H
#define FLC_CORECLASS_H

#include <falcon/setup.h>
#include <falcon/symbol.h>
#include <falcon/proptable.h>
#include <falcon/mempool.h>
#include <stdlib.h>

#define OVERRIDE_OP_NEG       "__neg"

#define OVERRIDE_OP_ADD       "__add"
#define OVERRIDE_OP_SUB       "__sub"
#define OVERRIDE_OP_MUL       "__mul"
#define OVERRIDE_OP_DIV       "__div"
#define OVERRIDE_OP_MOD       "__mod"
#define OVERRIDE_OP_POW       "__pow"

#define OVERRIDE_OP_AADD      "__aadd"
#define OVERRIDE_OP_ASUB      "__asub"
#define OVERRIDE_OP_AMUL      "__amul"
#define OVERRIDE_OP_ADIV      "__adiv"
#define OVERRIDE_OP_AMOD      "__amod"
#define OVERRIDE_OP_APOW      "__apow"

#define OVERRIDE_OP_INC       "__inc"
#define OVERRIDE_OP_DEC       "__dec"
#define OVERRIDE_OP_INCPOST   "__incpost"
#define OVERRIDE_OP_DECPOST   "__decpost"

#define OVERRIDE_OP_CALL      "__call"

#define OVERRIDE_OP_GETINDEX  "__getIndex"
#define OVERRIDE_OP_SETINDEX  "__setIndex"
#define OVERRIDE_OP_GETPROP   "__getProperty"
#define OVERRIDE_OP_SETPROP   "__setProperty"

#define OVERRIDE_OP_COMPARE   "__compare"
#define OVERRIDE_OP_ISTRUE    "__isTrue"
#define OVERRIDE_OP_IN        "__in"
#define OVERRIDE_OP_PROVIDES  "__provides"

namespace Falcon {

class VMachine;
class ItemDict;

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

   /** Locally cached factory. */
   ObjectFactory m_factory;

   /** Dictionary of dictionaries of states. */
   ItemDict* m_states;

   /** Shortcut for the init state to speed up instance creation. */
   ItemDict* m_initState;

   bool m_bHasInitEnter;

public:

   /** Creates an item representation of a live class.
      The representation of the class needs a bit of extra informations that are provided by the
      virtual machine, other than the symbol that generated this item.
      The module id is useful as this object often refers to its module in the VM. Having the ID
      recorded here prevents the need to search for the live ID in the VM at critical times.
   */
   CoreClass( const Symbol *sym, LiveModule *lmod, PropertyTable *pt );
   ~CoreClass();

   LiveModule *liveModule() const { return m_lmod; }
   const Symbol *symbol() const { return m_sym; }

   PropertyTable &properties() { return *m_properties; }
   const PropertyTable &properties() const { return *m_properties; }

   ObjectFactory factory() const { return m_factory; }
   void factory( ObjectFactory f ) { m_factory = f; }

   /** Creates an instance of this class.
      The returned object and all its properties are stored in the same memory pool
      from which this instance is created.
      On a multithreading application, this method can only be called from inside the
      thread that is running the VM.

      In some cases (e.g. de-serialization) the caller may wish not to have
      the object initialized and filled with attributes. The optional
      appedAtribs parameter may be passed false to have the caller to
      fill the instance with startup data.

      \note This function never calls the constructor of the object.
      \param user_data pre-allocated user data.
      \param bDeserialize set to true if you are deserializing an instance (that is, if its
         constructor should not configure it).
      \return The new instance of the Core Object.
   */
   CoreObject *createInstance( void *user_data=0, bool bDeserialize = false ) const;

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


   /** Returns true if the class is derived from a given symbol.
      This method checks if this class is compatible with the given symbol,
      or in other words, if the given symbol is present somewhere in the class
      hierarcy.

      True is returned also if this class is exactly created from the given
      symbol.

      \param sym The symbol that has to be checked for parentship.
      \return true if the class is derived from a class having the given name
   */
   bool derivedFrom( const Symbol* sym ) const;

   /** Marks the class and its inner data.
      This marks the class, the livemodule it is bound to, the property table data
      and the ancestors.
   */
   void gcMark( uint32 mark );

   /** Sets a state dictionary for this class.
    States are usually string -> CoreFunc dictionaries;
    so, an ItemDict like this will contain a set of String -> CoreDict( string -> CoreFunc );

    This is mainly used by VMachine::link.
    \note If a previous state dictionary was set, it will be destroyed.
    \param sd State dictionary
    \param is Dictionary for the init state, if existing.
   */

   void states( ItemDict* sd, ItemDict* is = 0 );
   ItemDict* states() const { return m_states; }

   ItemDict* initState() const { return m_initState; }
   bool hasInitEnter() const { return m_bHasInitEnter; }


   virtual void __neg( Item& result );

   virtual void __add( const Item& operand, Item& result );
   virtual void __sub( const Item& operand, Item& result );
   virtual void __mul( const Item& operand, Item& result );
   virtual void __div( const Item& operand, Item& result );
   virtual void __mod( const Item& operand, Item& result );
   virtual void __pow( const Item& operand, Item& result );

   virtual void __aadd( const Item& operand );
   virtual void __asub( const Item& operand );
   virtual void __amul( const Item& operand );
   virtual void __adiv( const Item& operand );
   virtual void __amod( const Item& operand );
   virtual void __apow( const Item& operand );

   virtual void __inc();
   virtual void __dec();
   virtual void __incpost();
   virtual void __decpost();

   virtual void __call( VMachine *vm );

   virtual bool __getIndex(const Item& index, Item& value );
   virtual bool __setIndex(const Item& index, const Item& value );

   virtual bool __getProperty( const String& prop, Item& value );
   virtual void __setProperty( const String& prop, const Item& value );

   virtual int __compare( const Item& value );
   virtual bool __isTrue();
   virtual bool __in( const Item& element );
   virtual bool __provides( const String& prop );
};

}

#endif

/* end of flc_cclass.h */
