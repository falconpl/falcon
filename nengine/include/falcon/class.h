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
class DataReader;
class DataWriter;

/** Representation of classes, that is item types.

 In falcon, each item has a type, which refers to a class.

 The Class represents the operations that can be preformed
 on a certain instance.
 
 To publish an item to the Virtual Machine, the calling program
 must create a class that instructs the VM about how the items
 of that type must be handled.
 
 Falcon::Class instances take care of the creation of objects, of their serialization
 and of their disposal. It is also responsible to check for properties

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
   virtual void serialize( DataWriter* stream, void* self ) const = 0;

   /** Deserializes an instance.
      The new instance must be initialized and ready to be "selfed".
   */
   virtual void* deserialize( DataReader* stream ) const = 0;

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
     @param self The object for which the properties have been requested.
     @param cb A callback function receiving one property at a time.

    The method may be called with a "self" 0 if this class has invariant
    instances.
     @note This base class implementation does nothing.
    */
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;


   /** Returns true if the class is derived from the given class
      \param className the name of a possibly parent class
      \return true if the class is derived from a class having the given name

      This function scans the property table of the class (template properties)
      for an item with the given name, and if that item exists and it's a class
      item, then this method returns true.
   */
   bool derivedFrom( Class* other ) const;

   /** Return true if the class provides the given property.
      @param self The object for which the properties have been requested.

      The method may be called with a "self" 0 if this class has invariant
      instances.
    */
   virtual bool hasProperty( void* self, const String& prop ) const;

   /** Return a summary or description of an instance.
    Differently from toString, this method ignores string rendering
    that may involve the virtual machine (i.e. toString() overloads).
 
    \note To be considered a debug device.
    */
   virtual void describe( void* instance, String& target ) const;

   //=========================================================
   // Operators.
   //

   /** Called back when the VM wants to negate an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original operand (where the instasnce was stored).
     \param target where to place the result of the operation.

    */
   virtual void op_neg( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to add something.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 The original second operand.
     \param target where to place the result of the operation.

    */
   virtual void op_add( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to subtract something.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 The original second operand.
     \param target where to place the result of the operation.

    */
   virtual void op_sub( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to multiply something.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 The original second operand.
     \param target where to place the result of the operation.

    */
   virtual void op_mul( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to divide something.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 The original second operand.
     \param target where to place the result of the operation.

    */
   virtual void op_div( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to apply the modulo operator.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 The original second operand.
     \param target where to place the result of the operation.

    */
   virtual void op_mod( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to apply the power operator on.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 The original second operand.
     \param target where to place the result of the operation.

    */
   virtual void op_pow( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to add something to an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
     \param op2 The original second operand.
    */
   virtual void op_aadd( VMachine *vm, void* self) const;

   /** Called back when the VM wants to subtract something to an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
     \param op2 The original second operand.
    */
   virtual void op_asub( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to multiply something to an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
     \param op2 The original second operand.
    */
   virtual void op_amul( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to divide something to an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
     \param op2 The original second operand.
    */
   virtual void op_adiv( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to apply the modulo something to an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
     \param op2 The original second operand.
    */
   virtual void op_amod( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to apply get the power of an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
     \param op2 The original second operand.
   */
   virtual void op_apow( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to increment (prefix) an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param operand The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
   */
   virtual void op_inc(VMachine *vm, void* self ) const;

   /* Called back when the VM wants to decrement (prefix) an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param operand The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
   */
   virtual void op_dec(VMachine *vm, void* self) const;

   /** Called back when the VM wants to increment (postfix) an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param operand The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
   */
   virtual void op_incpost(VMachine *vm, void* self ) const;
   
   /** Called back when the VM wants to decrement (postfix) an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param operand The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
   */
   virtual void op_decpost(VMachine *vm, void* self ) const;

   /** Called back when the VM wants to get an index out of an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 Index that must be accessed in op1.
     \param target where to store the result of the operation.
   */
   virtual void op_getIndex(VMachine *vm, void* self ) const;
   
   /** Called back when the VM wants to get an index out of an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 Index that must be accessed in op1.
     \param target where to store the result of the operation.
    
     \note normally, the result of a setindex operation is the same value that
    was set (op2).
   */
   virtual void op_setIndex(VMachine *vm, void* self ) const;

   /** Called back when the VM wants to get the value of a property of an item
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 Index that must be accessed in op1.
     \param target where to store the result of the operation.

     \note normally, the result of a setindex operation is the same value that
    was set (op2).
   */
   virtual void op_getProperty( VMachine *vm, void* self, const String& prop) const;

   /** Called back when the VM wants to set a value of a property in an item.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 Index that must be accessed in op1.
     \param target where to store the result of the operation.

     \note normally, the result of a property operation is the same value that
    was set (op2).
   */
   virtual void op_setProperty( VMachine *vm, void* self, const String& prop ) const;

   /** Called back when the VM wants to compare an item to this instance. 
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param op2 The compare paragon.
     \param target The place where to store the result.


    This method is called back on any comparison operation, including the six
    basic comparison operators (<, >, <=, >=, == and !=) but also in other
    ordering operations thar involve the virtual machine.

    The result placed in target should be
    0 if the two items are considered identical, < 0 if op1 is smaller
      than op2, > 0 if op1 is greater than op2.

    All the subclasses should call the base class op_compare as a residual
    criterion.
    */
   virtual void op_compare( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to know if an item is true.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored).
     \param target Where to store the result.

    This method is called back when the VM wants to know if an item can be
    considered true.

    The result placed in target should be a boolean true or false value.
    */
   virtual void op_isTrue( VMachine *vm, void* self ) const;

   /** Called back when the VM wants to know if an item is true.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The container (where the instasnce was stored).
     \param op2 The the item that must be searched.
     \param target Where to store the result.

    This method is called back when the VM wants to know if an item can be
    considered true.

    The result placed in target should be a boolean true or false value.
    */
   virtual void op_in( VMachine *vm, void* self ) const;
   
   /** Called back when the vm wants to know if a certain item provides a certain property.
     \param vm the virtual machine that will receive the result.
     \param self the instance in op1 (or 0 on flat items)
     \param op1 The container (where the instasnce was stored).
     \param op2 The the item that must be searched.
     \param target Where to store the result.

    This method is called back when the VM wants to know if an item can be
    considered true.

    The result placed in target should be a boolean true or false value.
    */
   virtual void op_provides( VMachine *vm, void* self ) const;

   /** Call the instance.
    \param vm A virutal machine where the call is performed.
    \param self An instance of this class
    \param pcount the number of parameters in the call.
    \param target The return value of the call.

    In case calling the given instance can be immediately done and is granted to
    be flat, this method can put the return value associated with this call in
    the \b target parameter (which is usually the top of the stack in this
    context.

    \note The default operation associated with the base class is that to raise
    a non-callable exception.
    \note The vm->self() item will always be the item where the self instance
    is stored.
    */
    virtual void op_call( VMachine *vm, int32 paramCount, void* self ) const;


   /** Implents textification operator for the Virtual Macine.
    \param vm the virtual machine that will receive the result.
    \param self the instance (or 0 on flat items)
    \param op1 The original item (where the instasnce was stored).
    \param target where to place the result of the operation.

    This method obtains immediately a textual value for the instance
    to be stored in the virtual machine, or prepares the call for the proper
    string geneartor code.

    The base class behavior is that of calling Class::describe() on the
    instance passed as self in the virtual machine, and then store it in
    VMachine::regA() as a garbageable string.
    */
   virtual void op_toString( VMachine *vm, void* self ) const;

protected:
   String m_name;
   int64 m_typeID;
   bool m_quasiFlat;
};

}

#endif

/* end of class.h */
