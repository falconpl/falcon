/*
   FALCON - The Falcon Programming Language.
   FILE: cclass.h

   Core Class definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Jan 2011 15:01:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core Class definition
*/

#ifndef FLC_CORECLASS_H
#define FLC_CORECLASS_H

#include <falcon/setup.h>
#include <falcon/item.h>

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

class FALCON_DYN_CLASS CoreClass
{
public:

   /** Creates an item representation of a live class.
      The representation of the class needs a bit of extra informations that are provided by the
      virtual machine, other than the symbol that generated this item.
      The module id is useful as this object often refers to its module in the VM. Having the ID
      recorded here prevents the need to search for the live ID in the VM at critical times.
   */
   CoreClass( );
   ~CoreClass();


   /** Returns true if the class is derived from a class with the given name.
      This function scans the property table of the class (template properties)
      for an item with the given name, and if that item exists and it's a class
      item, then this method returns true.
      \param className the name of a possibly parent class
      \return true if the class is derived from a class having the given name
   */
   bool derivedFrom( const String &className ) const;


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
