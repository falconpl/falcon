/*
   FALCON - The Falcon Programming Language.
   FILE: class.h

   Class definition of a Falcon Class
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

#ifndef FALCON_CLASS_H
#define FALCON_CLASS_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/enumerator.h>

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
class Item;

/** Representation of classes, that is item types.

 * In falcon, each item has a type, which refers to a class.
 * The CoreClass represents the operations that can be preformed
 * on a certain instance.
 *
 * To publish an item to the Virtual Machine, the calling program
 * must create a class that instructs the VM about how the items
 * of that type must be handled.
 *
 * CoreClasses take care of the creation of objects, of their serialization
 * and of their dispsal. It is also responsible to check for properties

 *
*/

class FALCON_DYN_CLASS Class
{
public:

   /** Creates an item representation of a live class.
      The representation of the class needs a bit of extra informations that are provided by the
      virtual machine, other than the symbol that generated this item.
      The module id is useful as this object often refers to its module in the VM. Having the ID
      recorded here prevents the need to search for the live ID in the VM at critical times.

       By default the TypeID of a class is -1, meaning undefined. Untyped class instances
       require direct management through the owning core class, while typed class instances
       can use their type ID to make some reasoning about their class.
    *
    *  Falcon system uses this number in quasi-type classes, as arrays or dictionaries,
    * but the ID is available also for user application, starting from the baseUserID number.
    */
   Class( const String& name );
   
   /** Creates a class defining a type ID*/
   Class( const String& name, int64 tid );
   
   virtual ~Class();

   const static int64 baseUserID = 100;

   const int64 typeID() const { return m_typeID; }
   const String& name() const { return m_name; }
   bool isFlat() const { return m_quasiFlat; }

   //=========================================
   // Instance management

   /** Creates an instance.
        @param creationParams A void* to data that can be used by the subclasses to initialize the instances.
    *   @return The instance pointer.
    * The returned instance must be ready to be put in the target object.
    */
   virtual void* create(void* creationParams=0 ) const = 0;

   /** Disposes an instance */
   virtual void dispose( void* self ) const = 0;

   /** Clones an instance */
   virtual void* clone( void* source ) const = 0;

   /** Serializes an instance */
   virtual void serialize( Stream* stream, void* self ) const = 0;

   /** Deserializes an instance.
      The new instance must be initialized and ready to be "selfed".
   */
   virtual void* deserialize( Stream* stream ) const = 0;
   
   //=========================================================
   // Class management
   //

   /** Marks an instance.
    The base version does nothing.
    */
   virtual void gcMark( void* self, uint32 mark ) const;

   /** Callback receiving all the properties in this class. */
   typedef Enumerator<String> PropertyEnumerator;

   /** List the properties in this class.
    * @param cb A callback function receiving one property at a time.
    *
    * @note This base class implementation does nothing.
    */
   virtual void enumerateProperties( PropertyEnumerator& cb ) const;


   /** Returns true if the class is derived from the given class
      \param className the name of a possibly parent class
      \return true if the class is derived from a class having the given name

      This function scans the property table of the class (template properties)
      for an item with the given name, and if that item exists and it's a class
      item, then this method returns true.
   */
   bool derivedFrom( Class* other ) const;

   /** Return true if the class provides the given property.
    */
   virtual bool hasProperty( const String& prop ) const;

   /** Return a summary or description of an instance.
    Differently from toString, this method ignores string rendering
    * that may involve the virtual machine (i.e. toString() overloads).
    *
    * To be considered a debug device.
    */
   virtual void describe( void* instance, String& target ) const;

   /** Notify about an assignemnt.
    @param instance The instance being assigned.
    @return the same instance, or a new instance if the assignment must be considered flat.

    Whenever the engine assigns an instance to a new item via a "=" expression,
    this method is called back. This gives the class the chance to alter the item,
    i.e. "colouring" it, or to return a new copy of the object if this is
    consistent with the assignment model.
    
    */
   virtual void* assign( void* instance ) const;
   //=========================================================
   // Operators.
   //

   virtual void neg( VMachine *vm ) const;

   virtual void add( VMachine *vm ) const;
   virtual void sub( VMachine *vm ) const;
   virtual void mul( VMachine *vm ) const;
   virtual void div( VMachine *vm ) const;
   virtual void mod( VMachine *vm ) const;
   virtual void pow( VMachine *vm ) const;

   virtual void aadd( VMachine *vm ) const;
   virtual void asub( VMachine *vm ) const;
   virtual void amul( VMachine *vm ) const;
   virtual void adiv( VMachine *vm ) const;
   virtual void amod( VMachine *vm ) const;
   virtual void apow( VMachine *vm ) const;

   virtual void inc(VMachine *vm ) const;
   virtual void dec(VMachine *vm ) const;
   virtual void incpost(VMachine *vm ) const;
   virtual void decpost(VMachine *vm ) const;

   virtual void call( VMachine *vm, int32 paramCount ) const;

   virtual bool getIndex(VMachine *vm ) const;
   virtual bool setIndex(VMachine *vm ) const;

   virtual bool getProperty( VMachine *vm ) const;
   virtual void setProperty( VMachine *vm ) const;

   virtual int compare( VMachine *vm )const;
   virtual bool isTrue( VMachine *vm ) const;
   virtual bool in( VMachine *vm ) const;
   virtual bool provides( VMachine *vm ) const;

   virtual bool toString( VMachine *vm ) const;

protected:
   String m_name;
   int64 m_typeID;
   bool m_quasiFlat;
};

}

#endif

/* end of class.h */
