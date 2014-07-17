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
#include <falcon/itemid.h>
#include <falcon/mantra.h>

namespace Falcon {

class VMContext;
class Item;
class DataReader;
class DataWriter;
class Module;
class ExprInherit;
class ItemArray;
class Error;
class Function;
class TextWriter;
class Selectable;

#define FALCON_CLASS_CREATE_AT_INIT          ((void*)1)
#define FALCON_CLASS_NO_NEED_FOR_INSTANCE    ((void*)2)

/** Representation of classes, that is item types.

 In falcon, each item has a type, which refers to a class.

 The Class represents the operations that can be preformed
 on a certain instance.

 To publish an item to the Virtual Machine, the calling program
 must create a class that instructs the VM about how the items
 of that type must be handled.

 Falcon::Class instances take care of the creation of objects, of their serialization
 and of their disposal. It is also responsible to check for properties

 You may wish to use the OpToken class to simplify the writing of operators in
 subclasses.

 \section implementing_operators Implementing operators

 At Falcon language level, operator overload is performed by specializing
 the operator methods in this class (the methods named op_*).

 All this methods have a prototype modelled after the following:

 @code
 virtual void op_something( VMContext* ctx, void* instance );
 @endcode

 Each operator receives an instance of the real object that this Class
 manipulates in the \b instance pointer and the virtual machine that requested
 the operation to be performed.

 Other than this, operands have also some parameters that are Item instances
 buried in the virtual machine stack in the moment they are called. Instead
 of unrolling this stack and passing the operand items to the method,
 it's safer and more efficient to keep them in the stack and let the operators
 to access their operands.

 There are several means to do that.

 The operator implementor may use the
 OpToken class that takes care of automaticly get the correct operand and
 unroll the stack correctly at the end of the operation.

 When maximum performance is needed, the VMachine::operands to get the operands
 and VMachine::stackResult to set a result for this operation.

 Finally, it is possible to access the operands directly from the current
 context stack.

 At low level, each operator MUST pop all its operands and MUST push exactly
 one result (or, pop all its operands - 1 and set the operand left in the stack
 as the result of the operation).

 Operators may go deep, calling the VMachine or other code that may in turn
 call the VMachine. In that case, the stack clean step should be performed
 by the deep step, before exiting the code frame.

 \note Usually the first operand is also the item holding the \b instance instance.
 However, instance is always passed to the operands because the VM did already some
 work in checking the kind of item stored as the first operand. However, the
 callee may be interested in accessing the first operand as well, as the Item
 in that place may hold some information (flags, out-of-band, copy marker and
 so on) that the implementor may find useful.

 Notice that \b instance is set to 0 when the class is a flat item reflection core
 class (as the Integer, Nil, Boolean and Numeric handlers). It receives a value
 only if the first operand is a FLC_USER_ITEM or FLC_DEEP_ITEM.

 \see OpToken

 \section class_serialize Item serialization
 TODO

 \section class_tostring Item representation

 Falcon items have different forms of representation, there is not a single way
 to turn an item in a string for output, as different forms have different intent.

 - Stringification (or toString()): this process turns an instance in a string
   for presentation to final-users. It's driven by the op_toString VM operator,
   and this allows the item itself to have a say in how the object is converted
   to a string. The operation is automatically invoked when an instance is
   added to a string. By default, a stringified instance just reports the class
   name.

 - description: This is a developer-oriented representation of the instance,
   recursively rendering the name of the instance class and the values of
   the public properties. If detected, methods are not included in the
   representation. This representation is invoked via the describe() method.

  - inspection: This describes the structure of a class instance. Public properties
    are recursively inspected, private properties and methods are just listed.

  - Rendering: this converts the instance to a representation suitable to be
  parsed as Falcon source code back into the rendered instance. Not all the instances
  are suitable for rendering (in which case, it's legal to throw an exception). Mostly,
  TreeStep subclasses and simple data types only are rendered.


*/

class FALCON_DYN_CLASS Class: public Mantra
{
public:
   Class( const String& name );
   Class( const String& name, int64 tid );
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
   Class( const String& name, Module* module, int line, int chr );

   /** Creates a class defining a type ID
    \param name The name of the class.
    \param the Type ID of the items created by this class.

    This constructor creates a class that is able to manage basic
    Falcon types. The TID associated with this is the numeric type
    ID of the datatype that are created as instances.

    Strings, Arrays, Dictionaries, Ranges and even Integer or NIL are
    all item types that have a class offering a TID.
    */
   Class( const String& name, int64 tid, Module* module, int line, int chr );
   virtual ~Class();

   const static int64 baseUserID = 100;

   int64 typeID() const { return m_typeID; }
   void typeID(int64 t) { m_typeID = t; }
   const String& name() const { return m_name; }
   /** Renames the class
    \param name The new name of the class.
    \note Will throw an assert if alredy stored in a module.
    */
   void name( const String& n ) { m_name = n; }

   /** Return true if this class can safely cast to a FalconClass class.
    \return true if this is FalconClass.

    Falcon classes have special significance to the engine, as they
    implement the Falcon OOP model, and can be used to build inheritance directly
    at syntactic level.

    This flag is true for FalconClass instances and (eventually) derived classes.
    */
   bool isFalconClass() const { return m_bIsFalconClass; }

   /**
    * Render the class back as source code.
    */
   virtual void render( TextWriter* tw, int32 depth ) const;


   /** Returns true if this class has flat instances.
    Flat instances are completely stored in the item accompaining the class.
    Their data is the (possibly volatile and transient) pointer to the item
    containing the whole information needed to rebuild the object.

    Flat classes known by the engine are:
    - ClassNil
    - ClassBool
    - ClassInteger
    - ClassNumeric
    - ClassMethod

    */
   bool isFlatInstance() const { return m_bIsFlatInstance; }

   /** Flag to check for the metaclass.
    The MetaClass is a special class that handle other classes.
    This resolves in typeID() == FLC_CLASS_ID_CLASS
    */
   bool isMetaClass() const { return typeID() == FLC_CLASS_ID_CLASS; }

   /** Determines if this is a class in the ErrorClass hierarcy.
    \return true if this is an error class.

    This flags allows to easily unbox error raised by scripts autonomously
    out of their handler class and treat them as proper Falcon::Error classes
    at C++ level.

    \note Theoretically, it is also possible to check Class::isDerivedFrom on the
    base Error* class provided by the StdErrors class in the engine, but
    this way is faster.
    */
   bool isErrorClass() const { return m_bIsErrorClass; }

   /** Gets a direct parent class of this class by name.
    \param name The name of the direct parent class.
    \return A class if the name represents a parent class, 0 otherwise.
    If the given class is among the parent list, this method returns the parent.

    The default behavior is that of always returning 0.
    */
   virtual const Class* getParent( const String& name ) const;

   /** Check if the given class is derived from this class.
    \param cls The possibly base class.
    \return True if cls is one of the base classes.

    The method returns true if the class is the same as this class or
    if one of the base classes of this class is derived from cls.

    The base implementation checks if the parameter is the same as \b this.
    Subclasses should check against theoretical or structured inheritance.
    */
   virtual bool isDerivedFrom( const Class* cls ) const;

   typedef Enumerator<const Class* > ClassEnumerator;

   virtual void enumerateParents( ClassEnumerator& cb ) const;

   /** Identifies the data known in a parent of a composed class.
    \param parent The parent of this class.
    \param data The data associated with this class.
    \return A pointer to a data that is known to the parent, or 0 if parent is
    unknown.

    Composed classes (for instance, HyperClass and Prototype) carry multiple
    instances representing the data which is related to a specific parents.
    This method can be used to retrieve the data which is associated with a
    particular subclass.

    If \b parent is this same class, then \b data is returned. Otherwise, if
    it's identified as a component of this class, an usable data is returned,
    while if \b parent is unknown, 0 is returned.

    In some contexts, parents might use the same data as their child; it's
    the case of incremental classes as FalconClass. In that case, \b data may be
    returned even if \b parent is a proper parent of this class.
    */
   virtual void* getParentData( const Class* parent, void* data ) const;


   //=========================================
   // Instance management

   /** Return a possibly accurate esteem of the memory used by this instance.
    *
    * By default, this method returns 0
    */
   virtual int64 occupiedMemory( void* instance ) const;

   /** Disposes an instance.
    \param instance The instance to be disposed of.
    \note Actually, the disposal might be a dereference if necessary.
    
    \note Flat instance classes need not to do anything.
    */
   virtual void dispose( void* instance ) const = 0;

   /** Clones an instance.
    \param instance The instance to be cloned.
    \return A new copy of the instance.
    
    \note Flat instance classes should return the instance parameter as-is.
    */
   virtual void* clone( void* instance ) const = 0;
   
  /** Creates a totally unconfigured instance of this class.
    \return a new instance of an entity of this class, 0 if instances
    cannot be created by scripts or FALCON_CLASS_CREATE_AT_INIT if the instance
    is created by the class during op_init invocation.
    
    This method creates an instance that is then to be initialized
    by other means (mainly, by a subsequent call to op_init).
    
    Some class hierarcies provide a "top-class" instance
    
    Some classes just provide static members and are not meant
    to generate any instance. It is then legal to return 0 
    (the system will generate a sensible error depending on the
    context).
   
   This function is not required to handle the created instance to the
   garbage collector. It is supposed that, in case this is reasonable,
   the garbage collector is invoked by the op_init() class member, or
   directly by the virtual machine when necessary.
   
   Once returned, the instance might be stored in the garbage collector
   by the virtual machine. This ensures that any subsequent operation during
   even complex init phases involving foreign subclasses can be correctly
   handled and that the VM knows how to handled the (not fully configured)
   instance during the following period.
   
   If the class cannot create the instance without the parameters that
   are passed to op_init, it can return FALCON_CLASS_CREATE_AT_INIT. If it does so,
   the class must:
   - create the instance
   - store it with stackResult or in the data stack item before the first parameter
   - store it in the garbage collector, if needed.


    \note Flat instance classes \b must return 0.
    */
   virtual void* createInstance() const = 0; 

   /** Store an instance to a determined storage media.
    @param ctx A virtual machine context where the serialization occours.
    @param stream The data writer where the instance is being stored.
    @param instance The instance that must be serialized.
    @throw IoError on i/o error during serialization.
    @throw UnserializableError if the class doesn't provide a class-specific
    serialization.

    By default, the base class raises a UnserializableError, indicating that
    there aren't enough information to store the live item on the stream.
    Subclasses must reimplement this method doing something sensible.

    @note the store is allowed to push PSteps to complete the storage at a
    later time. As the storage operation is complete, the context stack MUST be
    in the same status as when the storage begins.

    @see class_serialize
    */
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;

   /** Restores an instance previously stored on a stream.
    \param ctx A virtual machine context where the deserialization occurs.
    \param stream The data writer where the instance is being stored.
    \param empty A pointer that will be filled with the new instance (but see below)
    \throw IoError on i/o error during serialization.
    \throw UnserializableError if the class doesn't provide a class-specific
    serialization.

    On return restored item MUST be placed on top of the context data stack, or
    an exception MUST be thrown. Alternatively, the restore method may push
    a PStep on the code stack to complete the item restoring at a later stage. In any
    case, as the restore operation is completed, a single restored item MUST be pushed
    on top of the context data stack.

    By default, the base class raises a UnserializableError, indicating that
    there aren't enough information to store the live item on the stream.
    Subclasses must reimplement this method doing something sensible.

    \see class_serialize
   */
   virtual void restore( VMContext* ctx, DataReader* stream ) const;

   /** Called before storage to declare some other items that should be serialized.
    \param ctx A virtual machine context where the deserialization occurs.
    \param stream The data writer where the instance is being stored.
    \param instance The instance that must be serialized.

    This method is invoked before calling store(), to give a change to the class
    to declare some other items on which this instance is dependent.

    The subclasses should just fill the subItems array or eventually invoke the
    VM passing the subItem array to the called subroutine. The items that
    are stored in the array will be serialized afterwards.

    @note The subItems array garbage collection by the Storer instance that is
    controlling the serialization process. This means that the class can
    invoke other psteps and store the subItems array in a local variable or
    on the context stack without explicitly accounting for garbage.

    The base class does nothing.

    \see class_serialize
   */
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   /** Called after deserialization to restore other items.
    \param ctx A virtual machine context where the deserialization occours.
    \param stream The data writer where the instance is being stored.
    \param instance The instance that must be serialized.

    This method is invoked after calling restore(). The subItem array is
    filled with the same items, already deserialized, as it was filled by
    flatten() before serialization occured.

   @note The subItems array garbage collection by the Restorer instance that is
    controlling the deserialization process. This means that the class can
    invoke other psteps and store the subItems array in a local variable or
    on the context stack without explicitly accounting for garbage.

    The base class does nothing.

    \see class_serialize
   */
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual void delegate( void* instance, Item* target, const String& message ) const;

   //=========================================================
   // Class management
   //

   /** Marks an instance.
    \parm instance The instance of this class to be marked.
    \param mark The gc mark to be applied.

    This method is called every time an item with mark sign is inspected.

    The class has here the ability to set a mark flag on the instance instance,
    and eventually propagate the mark to other contents or data related
    to the instance instance.

    \note The base version does nothing.
    */
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;

   /** Determines if an instance should be disposed of.
    \parm instance The intance of this class to be marked.
    \param mark The current gc Mark.
    \return True if the instance is alive, false if it can be disposed.

    This method is invoked when the garbage collector wants to check if
    an instance is still alive or needs to be disposed. The class should
    check the incoming mark against the last mark that has been applied
    during a gcMark() call. If the \b mark parameter is bigger, then
    the object has been left behind and can be disposed at class will.

    Returning false, the garbage collector will free its own accounting
    resources and won't call the gcCheck() method anymore. The dispose
    method will be then called at a (near) future moment by the garbage collector
    when the object is finalized.

    Notice that disposing of the \b instance parameter without returning false
    might make the GC to present the same item again to this class, possibly
    causing crashes (double free, illegal memory access etc.).

    \note The base version does nothing.
    */
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   /** Callback receiving all the properties in this class. */
   typedef Enumerator<String> PropertyEnumerator;

   /** Emnumerate the properties in this class.
     @param instance The object for which the properties have been requested.
     @param cb A callback function receiving one property at a time.

    The method may be called with a "instance" 0 if this class has invariant
    instances.
     @note This base class implementation does nothing.
    */
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;

   /** Callback receiving all the properties with their values in this class. */
   class PVEnumerator
   {
   public:
      virtual void operator()( const String& property, Item& value ) = 0;
   };

   /** Emnumerate the properties in this class with their associated values.
     @param instance The object for which the properties have been requested.
     @param cb A callback function receiving one property at a time.

     @note This base class implementation does nothing.
    */
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;


   /** Return true if the class provides the given property.
      @param instance The object for which the properties have been requested.
      @param prop The property that is searched.

      The method may be called with a "instance" 0 if this class has invariant
      instances.
    */
   virtual bool hasProperty( void* instance, const String& prop ) const;

   /** Lists all the summonings accepted by this instance.
      @param instance The object for which the properties have been requested.
      @param cb An enumerator receiving all the messages to which this object responds.

      The method may be called with a "instance" 0 if this class has invariant
      instances.

      \note The default behavior of this method is to invoke enumerateProperties.
    */
   virtual void enumerateSummonings( void* instance, PropertyEnumerator& cb ) const;

   /** Return a (possibly deep) summary or description of an instance.
      \param instance The instance to be described.
      \param target Where to place the rendered string.
      \param depth Maximum depth in recursions.
      \param maxlen Maximum length.

    This method returns a string containing a representation of this item. If
    the item is deep (an array, an instance, a dictionary) the contents are
    also passed through this function.

    This method traverses arrays and items deeply; there isn't any protection
    against circular references, which may cause endless loop. However, the
    default maximum depth is 3, which is a good depth for debugging (goes deep,
    but doesn't dig beyond average interesting points). Set to -1 to
    have infinite depth.

    By default, only the first 60 characters of strings and elements of membufs
    are displayed. You may change this default by providing a maxLen parameter.

    \note Differently from toString, this method ignores string rendering
    that may involve the virtual machine. Even if describe() is defined as a
    method that might be invoked by the VM, that will be ignored in
    renderings of items in a container.
    */
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   /**
    * Describes the deep structure of the object.
    *
    * \note It is acceptable for this method to default to describe() or render().
    */
   virtual void inspect( void* instance, String& target, int depth = 3 ) const;

   /** Called back by an inheritance when it gets resolved.
    \param inh The inheritance that has been just resolved.

    When a foreign inheritance of a class gets resolved during the link
    phase, the parent need to know about this fact to prepare itinstance.

    The inheritance determines if the owner is a falcon class, and in case
    it is, it calls back this method.

    The default behavior of Class is empty. This is commonly used only in
    FalconClass.
    */
   virtual void onInheritanceResolved( ExprInherit* inh );
   
   //=========================================================
   // Operators.
   //

   /** Invoked by the VM to initialize an instance.
    \param ctx a virtual machine context invoking the object initialization.
    \param instance The instance to be initialized.
    \param pcount Number of parameters passed in the init request.
    \return True if the init operation requires to go deep.
    
    This method is invoked by the VM when it requires an instance to be
    initialized by this class. The VM grants that the self item that is 
    being initialized is pushed right before the first parameter, and 
    can be found at opcodeParams( pcount+1 ). Notice that op_init() is called
    also for base classes of the item currently being initialized, so the
    item at opcodeParams( pcount+1 ) can be an instance of a child class.
    
    A call to this method will usually follow a call
    to createInstance(), where the instance is created; prior calling this 
    method, the VM ensures that the created instance is stored in the
    garbage collector, so that any operation causing suspension, deep
    descent or error in the VM won't leave the instance dangling in the void,
    and can be accounted in case of deep and complex initialization rotuines.

    If the initialization fails for any reason, this method should throw
    an error.

    If the method needs to invoke other VM operations, it must return \b true.
    Doing so, it declares that the initialization sequence passes under
    the control of this class. This class must then push proper PSteps in the
    stack so that, when the initialization step is complete, \b pcount parameters
    are removed from the data stack.
    
    In case this method doesn't need to perform deep operations in the VMContext,
    it can simply return \b false. In this case, the caller will properly remove 
    items from the data stack as needed.
    
    In any case, pushing an instance of the class on the stack is \b not a
    responsibility of this method. This operation is performed by the caller,
    where needed.
    
    \note Flat instance classes will receive the item to be initialized
    in the \b instance parameter.
    
    */
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;

   /** Called back when the VM wants to negate an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is unary -- requires OpToken with 1 parameter.

    */
   virtual void op_neg( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to add something.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    */
   virtual void op_add( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to subtract something.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    */
   virtual void op_sub( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to multiply something.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    */
   virtual void op_mul( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to divide something.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    */
   virtual void op_div( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to apply the modulo operator.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    */
   virtual void op_mod( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to apply the power operator on.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    */
   virtual void op_pow( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to apply the shift right operator.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    */
   virtual void op_shr( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to apply the shift left operator.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    */
   virtual void op_shl( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to add something to an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.
    */
   virtual void op_aadd( VMContext* ctx, void* instance) const;

   /** Called back when the VM wants to subtract something to an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.
    */
   virtual void op_asub( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to multiply something to an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.
    */
   virtual void op_amul( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to divide something to an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.
    */
   virtual void op_adiv( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to apply the modulo something to an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance in op1 (or 0 on flat items)
     \param op1 The original first operand (where the instasnce was stored),
      that is also the place where to store the result.
     \param op2 The original second operand.
    */
   virtual void op_amod( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to get the power of an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.
   */
   virtual void op_apow( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to apply get the power of an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.
   */
   virtual void op_ashr( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to apply get the shift-left of an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.
   */
   virtual void op_ashl( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to increment (prefix) an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance in op1 (or 0 on flat items)
     \param instance the instance (or 0 on flat items)

    \note The operand is unary -- requires OpToken with 1 parameter.
   */
   virtual void op_inc( VMContext* ctx, void* instance ) const;

   /* Called back when the VM wants to decrement (prefix) an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance in op1 (or 0 on flat items)
     \param instance the instance (or 0 on flat items)

    \note The operand is unary -- requires OpToken with 1 parameter.
   */
   virtual void op_dec(VMContext* ctx, void* instance) const;

   /** Called back when the VM wants to increment (postfix) an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance in op1 (or 0 on flat items)
     \param instance the instance (or 0 on flat items)

    \note The operand is unary -- requires OpToken with 1 parameter.
   */
   virtual void op_incpost(VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to decrement (postfix) an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance in op1 (or 0 on flat items)
     \param instance the instance (or 0 on flat items)

    \note The operand is unary -- requires OpToken with 1 parameter.
   */
   virtual void op_decpost(VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to get an index out of an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    The first operand is instance, the second operand is the index to be accessed.
   */
   virtual void op_getIndex(VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to get an index out of an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is ternary -- requires OpToken with 3 parameters.

    Operands are in order:
    - The item holding this \b instance instance
    - The index being accessed (the value inside the [] brackets)
    - The new value for the item.

    Normally, the value of the operation should match with the value of the
    new item stored at the required index. For instance, the value of the
    following Falcon code:

    @code
    value = (array[10] = "something")
    @endcode

    is expected to be "something", but this method may set it to any sensible
    value the implementor wants to pass back.

   */
   virtual void op_setIndex(VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to get the value of a property of an item
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)
    \param prop The property to be accessed.

    \note The operand is unary -- requires OpToken with 1 parameter.
   */
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;

   /** Called back when the VM wants to set a value of a property in an item.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)
    \param prop The property to be accessed.

    This operator gets the following parameters in the stack:
       - TOP: self item
       - TOP-1 (VMContext::opcodeParam(1)): value to be stored
    
    It is supposed to remove this two items and leave the evalution of the expression
    a.x = b (usually b, the value) on the top of the stack. Thus, just removing the self
    item is usually what you're required to do.
   */
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop ) const;

   /** Called back when the VM wants to get the value of a property of this class (static property).
     \param ctx the virtual machine context that will receive the result.
    \param prop The property to be accessed.

    This operator gets the following parameters in the stack:
       - TOP: self item

    It is required to remove this item and put the value stroed in self.prop on top
    of the stack.
   */
   virtual void op_getClassProperty( VMContext* ctx, const String& prop) const;

   /** Called back when the VM wants to set a value of a property in an item.
     \param ctx the virtual machine context that will receive the result.
    \param prop The property to be accessed.

    \note The operand is binary -- requires OpToken with 2 parameters.
   */
   virtual void op_setClassProperty( VMContext* ctx, const String& prop ) const;

   /** Called back when the VM wants to compare an item to this instance.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note Signature: (1: Comparand) (0: self item) --> (0: result)

    This method is called back on any comparison operation, including the six
    basic comparison operators (<, >, <=, >=, == and !=) but also in other
    ordering operations thar involve the virtual machine.

    The result of the operation should be
    - 0 if the two items are considered identical,
    - \< 0 if op1 is smaller than op2,
    - \> 0 if op1 is greater than op2.

    All the subclasses should call the base class op_compare as a residual
    criterion.
    */
   virtual void op_compare( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to know if an item is true.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is unary -- requires OpToken with 1 parameter.

    This method is called back when the VM wants to know if an item can be
    considered true.

    The result placed in target should be a boolean true or false value.
    */
   virtual void op_isTrue( VMContext* ctx, void* instance ) const;

   /** Called back when the VM wants to know if an item is true.
     \param ctx the virtual machine context that will receive the result.
     \param instance the instance (or 0 on flat items)

    \note The operand is binary -- requires OpToken with 2 parameters.

    This method is called back when the VM wants to know if an item can be
    considered true.

    The result should be a boolean true or false value.
    */
   virtual void op_in( VMContext* ctx, void* instance ) const;

   virtual void op_provides( VMContext* ctx, void* instance, const String& propName ) const;


   /** Call the instance.
    \param ctx A virutal machine where the call is performed.
    \param instance An instance of this class
    \param pcount the number of parameters in the call.

    This operation has variable count of parameters. They are placed in reverse
    order on the stack. The item being called is always pushed before the
    parameters.

    When done, the operator should unroll the stack (VMContect::popData) of exactly
    paramCount elements; this will leave the called item on top of the stack.
    That item must be overwritten with the return value of this operation (might
    be nil).

    \note The default operation associated with the base class is that to raise
    a non-callable exception.
    */
   virtual void op_call( VMContext* ctx, int32 paramCount, void* instance ) const;

   /** Implements textification operator for the Virtual Macine.
    \param ctx the virtual machine context context that will receive the result.
    \param instance the instance.
    \note The operand is unary -- requires OpToken with 1 parameter.

    This method obtains immediately a textual value for the instance
    to be stored in the virtual machine, or prepares the call for the proper
    string geneartor code.

    The base class behavior is that of calling Class::describe() with depth 1
    and without maximum lenght on the instance passed as instance in the virtual
    machine and uses it as the result of the operation.

    Implementors not willing to use describe() or wishing to skip an extra
    virtual function call should reimplement this class.
    */
   virtual void op_toString( VMContext* ctx, void* instance ) const;

   /** Prepares iteration.
    \param ctx the virtual machine context context that will receive the result.
    \param instance the instance (or 0 on flat items)
    \note The operand unary, and generates an item.

    \b signature: (0: seq) -> (1: seq) (0: iter)

    This method call is generated by the virtual machine when it
    requires the start of an iteration on the contents this item.

    The sublcasses reimplementing this method should prepare a value
    that will be presented afterward to op_next.

    An item is left in the stack by the caller; op_iter should add a new item
    with an iterator that can then be used by op_next, or stored elsewhere.

    In case the class can't create an iterator, it should just push a nil item.
    */
   virtual void op_iter( VMContext* ctx, void* instance ) const;

   /** Continues iteration.
    \param ctx the virtual machine context context that will receive the result.
    \param instance the instance (or 0 on flat items)
    \note The operand is binary and leaves a new item in the stack.

    \b signature: (1: seq) (0: iter) --> (2: seq) (1: iter) (0: item|break)

    This method call is generated by the virtual machine when it
    requires the start of an iteration on the contents this item.

    This is always called after a sccessful op_iter to get an ready-to-be-used
    iterator in the collection.

    If the class is able to know that, it should post an item with the doubt
    bit set when the last nth item of a n-sized collection is retreived. When
    the collection is exausted, it should post a nil item with the break bit
    set at top of the stack.
    */
   virtual void op_next( VMContext* ctx, void* instance ) const;

   /** Summon an instance.
    \param ctx the virtual machine context context that will receive the result.
    \param instance the summoned instance (or 0 on flat items)
    \param message The message sent to the summoned object
    \param pCount number of parameters for the summoning
    \param bOptional True if the message is optional.

    \b signature: (pcount+1: instance) (pcount: parm0) ... (0:paramN) --> (0: result)

    This method call is generated by the virtual machine when an object is
    summoned.
    
    Summoning an instance is like getting a property or invoking a method, with
    three major differences:
    - The summon is an action that is performed on the object, not a request to the object
      to provide a property or method.
    - The object can propagate the summoning to other responders that will answer that call
     (delegation).
    - Summoning can be optional; in that case, the summoning operation is not performed (and
      the expression resolving as the summon parameters are not evaluated).

    \note The default behavior is that of returning or setting a property value if the summoned
          name corresponds to a property, or invoking a method if the summoned name is a method
          name.
    */
   virtual void op_summon( VMContext* ctx, void* instance, const String& message, int32 pCount, bool bOptional ) const;

   virtual void op_summon_failing( VMContext* ctx, void* instance, const String& message, int32 pCount ) const;

   //================================================================
   // Utilities
   //

   /** User flags.
    User flags are at disposal of the user to define some subtyping of class hierarcies.

    The ClassTreeStep hierarchy uses it to mark some special classes that have some
    significance to the engine.

    Although there's no mechanism to guarantee that the numbers and flags assigned
    are unique, if the user is reasonably safe about the usage domain of a certain
    class, it's safe to make assumption on the userFlags, which are never queried
    or altered by the engine.

    For instance, as TresSteps are bound to give off a certain classes which tightly
    relates to them, applying userFlags() on TreeStep::cls() return value is a
    "safe domain". A set of classes having a given parent module built by the user
    are another "safe domain".

    \note The base Class constructor zeroes the user flags.
    */
   int32 userFlags() const { return m_userFlags; }

   void userFlags( int32 uf ) { m_userFlags = uf; }
   
   /** Handler class for this class. */
   Class* handler() const;
   
   /** Creates an "automatic" child class.
    *
    * \param name The name of the child class
    * \param typeID an optional type ID for the class instances.
    *
    * This method is intended to create simple child classes out of
    * standard engine classes, or base classes provided in extension
    * modules, to be then configured via addMethod, addProperty and so on.
    *
    * The instance returned is an instance of ChildClass, which,
    * by default, has all the (relevant) virtual members overridden with
    * a method that calls that same member in the parent class (this one).
    *
    * For instance, the op_add operator of the child class will invoke
    * the same op_add of this class.
    *
    * By default, the new class takes the same type ID of the parent class.
    * Also, instance flatness and other relevant characteristics of the parent
    * class (this one) are copied.
    */
   Class* createChild(const String& name, int64 typeID=-1 ) const;

   /** Returns the destruction order of this class.
    *
    * Higher classes priorities are destroyed later when the collector performs a clear loop.
    * Classes handling normal items have priority 0 (the default).
    *
    * Subclasses handling other classes or instance which holds complex entities should set this
    * value according to the following scheme:
    * - 0: Generic instances with no priority issues (it's the default).
    * - 1: Mantras (generic functions)
    * - 2: Classes (In particular, Falcon classes)
    * - 3: Modules
    * - 4: Super modules (meta-modules, module spaces etc.)
    *
    * To configure the priority, set the m_clearPriority value in the child handler
    * class.
    */
   int clearPriority() const { return m_clearPriority; }

   typedef void (*setter)( const Class* cls, const String& name, void *instance, const Item& value );
   typedef void (*getter)( const Class* cls, const String& name, void *instance, Item& value );


   /**
      Adds a property to the class.
      \param name The name of hte property
      \param set A function invoked to set the property
      \param get A function invoked to get the property
      \param isHidden If true, the property is found when searched, but won't be 
             displayed in automatic rendering of the class.
      \param isStatic If true, the property is static (found on the class).

      If a property has no setter function it's considered read-only.
      You can also set the getter function to 0 to make it write-only; this will make it also automatically hidden.

      \note Existing properties having the same name will be overwritten.
   */
   void addProperty( const String& name, getter get, setter set=0, bool isStatic = false, bool isHidden = false );

   /** Adds a function to the class (as a method). 
      \param func The method code.
      \param isStatic If true, the method is considere static (to be applied to the class).      
      
      The given method function is owned by the class and will be disposed as the class is destroyed. 
      If this function returns false, the method is not actually added to the class.

      \note the name of the property is assumed to be the name declared by the Function instance.
      \note Existing properties having the same name will be overwritten.
   */
   void addMethod( Function* func, bool isStatic = false );

   /** Adds a function to the class (as a method). 
      \param name The name of the method to be added.
      \param func The method code.
      \param prototype The method prototype.
      \param isStatic If true, the method is considere static (to be applied to the class).
      \return true if the method can be added, false if the method was already present.
      
      This version takes a C function that should just push the desired result on top
      of the stack. 

      If the function returns false, the method is not actually added to the class.
      \note Function return (returnFrame()) is handled by the underlying Function
         code.
      \note Existing properties having the same name will be overwritten.
   */
   Function* addMethod( const String& name, ext_func_t func, const String& prototype, bool isStatic = false );

   /** Sets a function to be the constructor code (invoked at init).    
      \param Function the constructor function.

      \note The function is forcefully terminated as this invocation is complete.
      If more steps are required in the initialization of the class, the op_init() must
      be reimplemented.
   */
   void setConstuctor( Function* func );

   /** Sets a function to be the constructor code (invoked at init). 
      \param func the constructor function.
      \param prototype The constructor function prototype.
      
   */
   void setConstuctor( ext_func_t func, const String& prototype );

   /** Returns the constructor
    * \return the constructor
    */
   Function* getConstructor() const;

   /** Adds a class-wide constant. 
      \param name the name of the constant.
      \param value The value that the constant should assume.
      \return true if the name was free, false if it was already assigned.
      \note Existing properties having the same name will be overwritten.
   */
   void addConstant( const String& name, const Item& value );

   /** Sets the parent of this class.
      \param parent the parent class.

      This method is used to implement a simple single-parentship inheritance system.
      More sophisticated inheritance schemes require the subclasses to reimplement
      getParentData() and isDerivedFrom() methods.

      The parent class method should be able to handle the same type of entity
      handled by this child class. In other words, the structure or class handled by
      the methods in the parent class must be a super-structure or super-class of the
      data handled by this class. Again, more sophisticated approaches require the 
      class system that needs to be implemented to reimplement getParentData() and isDerivedFrom()
      methods.

      When an instance of this class is invoked, the parents constructors are \b not called.
      This class constructor must handle the initialization of the created object entirely.
      A different initialization scheme requires the classes involved to reimplement
      directly the op_init() method of this class.
   */
   void setParent( const Class* parent );

   /** GCMark this class.
   */
   virtual void gcMark( uint32 mark );

   /** True if this class has shared instances.
    The GC tracer (optionally activated in debug mode only) keeps track of every
    item stored in the GC, and throws an assert in case some error condition is
    detected.

    One of the error conditions is storing an already known data a second time
    in the GC, as this is probably an error that will cause a double-delete at
    a later time.

    However, some classes handle regularly multiple copies of the same instance
    to the GC, using some internal reference count or other mechanism to avoid
    just deleting the deep instance when the GC detects an unused item.

    If this class handles shared instances, the m_bHasSharedInstances member
    should be turned on by the subclass constructor;
    this method will then return true.
    */
    bool hasSharedInstances() const { return m_bHasSharedInstances; }

    /** Return the selectable interface for this instance.
     @param instance the instance that needs to be selected.
     @return A valid selectable entity or 0.

     This method is invoked by the selector system when an entity is fed into it.

     An instance that can be multiplexed by a selector through the Mupltiplex class
     will have a newly allocated selectable instance returned.

     The selectable entity is then held by the selector, which will refcount it and
     properly dispose of it, eventually gc-marking the instance when necessary.

     The default behavior is that of returning 0 (that means, the given instance
     is not selectable/doesn't offer a Multiplexing facility).
     */
    virtual Selectable* getSelectableInterface( void* instance ) const;

protected:
   bool m_bIsFalconClass;
   bool m_bIsErrorClass;
   bool m_bIsFlatInstance;
   bool m_bHasSharedInstances;

   /** This flags are at disposal of subclasses for special purpose (i.e. cast conversions). */
   int32 m_userFlags;

   int64 m_typeID;

   int32 m_clearPriority;

   const Class* m_parent;

private:

   class Private;
   Private* _p;
};

}

#endif

/* end of class.h */
