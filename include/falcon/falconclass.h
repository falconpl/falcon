/*
   FALCON - The Falcon Programming Language.
   FILE: falconclass.h

   Class defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 16:04:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FALCONCLASS_H_
#define _FALCON_FALCONCLASS_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/overridableclass.h>
#include <falcon/statement.h>
#include <falcon/synfunc.h>

namespace Falcon
{

class FalconInstance;
class Item;
class Function;
class DataReader;
class DataWriter;
class FalconState;
class Expression;
class MetaClass;
class HyperClass;
class VMContext;

class ExprInherit;
class ExprParentship;


/** Class defined by a Falcon script.

 This class implements a definition of a class in a falcon script.

 This structure is adequate to hold properties, methods, states, an init block
 and class parentship as they are declared in a script through the "class"
 declaration.

 Of course, it is possible to synthethize a FalconClass in a C++ extension or
 module; also, a method can be any Falcon function, including external
 functions. It is also possible to dynamically create new classes, or alter
 the methods and states of existing ones. However, extension code may find
 more convenient do derive its own data handler directly from the Class base
 class; multiple inheritance is anyhow possible also for direct Class
 implementations. Using a FalconClass is meaningful only when there is the need
 to create a class definition out of Falcon source code, or when the foreign
 code wants to exploit some abilities of Falcon source code (as i.e. type
 polimorphism, automatic content GC marking and so on).

 FalconClass instances are handled by the engine through the CoreClass handler.

 This class has also the necessary data to create "instances". A FalconInstance
 is an ordered array of values that match the properties in the class, and
 is handled through a CoreInstance handler, which reads the properties in the
 FalconClass that is associated with the instance.

 Falcon classes can have either an empty parentship or can be derived from
 other FalconClass entities. Deriving a script class from a parent which is not
 a FalconClass itself (or from multiple parents, where one of the parents
 is not a FalconClass is implemented by creating an HyperClass instance.

 It is possible to access "static" methods and base classes from a live
 script.

 Despite the fact that modules created from source Falcon scripts will
 always create FalconClass instances, the VM may generate an HyperClass in
 their place if one of their parent classes is found to be a non-FalconClass
 instance during the link process, while resolving external symbols.

 */
class FALCON_DYN_CLASS FalconClass: public OverridableClass
{
public:

   class Property
   {
   public:
      typedef union tag_cnt
      {
         Function* func;
         ExprInherit* inh;
         FalconState* state;
      } Value;

      typedef enum tag_proptype {
         t_prop,
         t_func,
         t_inh,
         t_state,
         t_static_prop,
         t_static_func
      } Type;

      String m_name;
      Type m_type;
      Value m_value;
      Item m_dflt;

      Property( const Property& other );
      Property( const Property& other, bool copyInitExpr );
      
      inline Property( const String& name, Type t ):
         m_name( name ),
         m_type(t),
         m_expr(0)
      {
      }
      
      inline Property( const String& name, const Item& value ):
         m_name( name ),
         m_type(t_prop),
         m_expr(0)
      {
         m_dflt = value;
      }

      Property( const String& name, const Item& value, Expression* expr );

      Property( Function* value, bool bIsStatic = false );
      Property( ExprInherit* value );
      Property( FalconState* value );

      ~Property();

      bool isStatic() const { return m_type == t_static_func || m_type == t_static_prop; }

      /** Returns associated expression (if this property holds an expression).
       \return a valid PCode or 0 if this is not an expression.
       */
      Expression* expression() const { return m_expr; }
      void expression(Expression* e) { m_expr = e; }

      Property* clone() const { return new Property(*this); }
   private:
      Expression* m_expr;
   };

   FalconClass( const String& name );
   virtual ~FalconClass();

   void* createInstance() const;
   
   /** Creates a new instance.
    \return A new instance created out of this class.
    The method will return 0 if the class is not completely resolved,
    i.e. if there is some parent that couldn't be found.
   */
   FalconInstance* createFalconInstance() const { 
      return static_cast<FalconInstance*>( createInstance() );  
   }

   /** Adds a property.
    \param name The name of the property to be added.
    \param initValue The initial value of the property.
    \return True if the property could be added, false if the name was already
    used.

    The initial value could be a DeepData instance; in this case, the
    class will provide GC support. However, it is preferable to store data
    that will stay valid as long as the class exists (i.e. as long as the
    module where the class is declared exists).

    \note In case this need arises, consider adding a UserData
    to a FalconClass subclass, and passing it as initValue here.
    */
   bool addProperty( const String& name, const Item& initValue, bool bIsStatic = false );

   /** Adds a property.
    \param name The name of the property to be added.
    \param initExpr An expression that must be invoked to initialize the property.
    \return True if the property could be added, false if the name was already
    used.

    The initial value could be a DeepData instance; in this case, the
    class will provide GC support. However, it is preferable to store data
    that will stay valid as long as the class exists (i.e. as long as the
    module where the class is declared exists).

    \note In case this need arises, consider adding a UserData
    to a FalconClass subclass, and passing it as initValue here.
    */
   bool addProperty( const String& name, Expression* initExpr );

    /** Adds a property.
    \param name The name of the property to be added.
    \return True if the property could be added, false if the name was already
    used.

    The initial value of the property is Nil.
    */
   bool addProperty( const String& name);

   /** Adds a method to this class.
    \param mth The method to be added.
    \return True if the method could be added, false if the name was already
    used.

    The methods in added are not property of the class; they are handled by
    the garbage collector, or the caller must be able to remove them when not
    needed anymore.
    */
   bool addMethod( Function* mth, bool bIsStatic = false );
   
   bool addMethod( const String& name, Function* mth, bool bIsStatic = false );

   bool hasInit() const { return m_hasInit; }
   void hasInit( bool bMode ) { m_hasInit = bMode; }

   /** Adds a parent and the parentship declaration.
    \param inh The inheritance declaration.
    \return true If the name of the remote class is free, false if it was
                 already assigned.

    The parentship delcaration carries also declaration for the parameters
    and the expressions needed to create the instances.

    The ownership of the inheritance records is held by this FalconClass.

    Inheritances may be initially unresolved; this is accounted by this FalconClass.
    When all the inheritances have been resolved through resolveInheritance,
    the class is automatically finalized (it's components are setup).
    */
   bool addParent( ExprInherit* inh );

   /** Sets the whole parentship structure. */   
   bool setParentship( ExprParentship* inh );

   /** Gets a member of this class.
    \param name The name of the member to be returned.
    \param target The item where the member will be stored.
    \return True if the member exists, false otherwise.
    A member of a class can be:
    - A property
    - A method
    - A base class.

    If a member with the given name exists, the \b target is filled with
    the correct value of the member (default value of the property,
    method or class) and the method returns true.

    If the name is not found, the method returns false.
    */
   bool getProperty( const String& name, Item& target ) const;

   /** Gets a member of this class -- as a definition.
    \param name The name of the member to be returned.
    \return A pointer to a property in this class or zero if not found.

    This method is mainly used by FalconInstance to get the member
    item in the class and then return a coherent data type to the VM
    that has accessed the item.
    */
   const Property* getProperty( const String& name ) const;

   /** Adds a state to this class.
      \param state The state to be added.
      \return True if the state could be added, false if the state name was
    already a member of this class.

      A state is a named set of methods that is dynamically substituted to the
      class.

    \note States are owned by this class. They are destroyed when this class
    is destroyed. However, methods in the states are not property of this
    class.

    \note A property named after the state is automatically added to the class.
    The property holds a State object (CoreState - FalconState value).
   */
   bool addState( FalconState* state );

   /** List the members in this class having property semantics.
     @param cb A callback function receiving one property at a time.

    This method enumerates the read-write items of this class. Some
    properties may contain a function, becoming actually a method,
    however they have a different semantic. They cannot be changed at
    class level and they cannot be subject to state changes.

    */
   void enumeratePropertiesOnly( PropertyEnumerator& cb ) const;

   virtual bool isDerivedFrom( const Class* cls ) const;
   virtual void* getParentData( const Class* parent, void* data ) const;

   //====================================================================
   // Overrides from Class
   //

   virtual Class* getParent( const String& name ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;

   //=========================================================
   // Class management
   //

   /** Marks the items in this class (if considered necessary).
    \param mark The mark that is used by GC.

    This is used to mark the default values of the class, if
    there is one or more deep item in the class.
    */
   virtual void gcMark( uint32 mark );

   virtual void gcMarkInstance( void* self, uint32 mark ) const;
   virtual bool gcCheckInstance( void* self, uint32 mark ) const;


   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;


   /** Creates the constructor or returns it if it's already here. */
   SynFunc* makeConstructor();

   /** Return the constructor of this class. */
   SynFunc* constructor() const { return m_constructor; }
   
   void setConstructor(SynFunc* sf ) { m_constructor = sf; }

   /** Create the class structure compiling it from its parents.
    \return false if there is still some unknown parent,
            true if the construction process is complete.

    This is called by the VM after all the missing parents have been
    found, provided that all the parents are declared as FalconClass subclasses.

    The role of this method is that to build the property list that composes
    this class, flattening the properties inherited from the base classes.
    
    A FalconClass needs to be constructed prior being used. Failing to do so
    will cause an error at instance creation.
    
    If some of the parents are not FalconClass, the engine must generate an
    hyperclass out of this falcon class.
    */
   bool construct( VMContext* ctx );

   /** Return the count of currently unknown parents.
    \return The count of unresolved inheritances.
    */
   int missingParents() const { return m_missingParents; }

   /** Check if all the declared inheritances are pure falcon classes.
    \return True if this class can be generated as a Falcon Class.

    */
   bool isPureFalcon() const { return m_bPureFalcon; }

   /** Construct this class as an hyperclass.
    \return an HyperClass representing this FalconClass.
    \note after this call, the returned hyperclass is the sole owner of this
          FalconClass instance. Every other reference to this FalconClass
          should be abandoned.
    */
   HyperClass* hyperConstruct();

   /** Pushes a VM step to initialize the instance properties.
    It expects to be called while inside a constructor frame.
    */
   void pushInitExprStep( VMContext* ctx );
   
   virtual Class* handler() const;
   
   /** Used by the module link system to prepare attribute generation for methods.
    *
    */
   bool registerAttributes( VMContext* ctx );

   void op_summon( VMContext* ctx, void* instance, const String& message, int32 pCount, bool bOptional ) const;
   void delegate( void* instance, Item* target, const String& message ) const;

   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   //=========================================================
   // Storer helpers
   //
   void storeSelf( DataWriter* wr ) const;
   void restoreSelf( DataReader* wr );
   
   void flattenSelf( ItemArray& flatArray ) const;
   void unflattenSelf( ItemArray& flatArray );
   
   /**
    * Render the class back as source code.
    */
   void render( TextWriter* tw, int32 depth )  const;

   //=========================================================
   // Operators.
   //

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;
   virtual void op_getClassProperty( VMContext* ctx, const String& prop) const;
   virtual void op_setClassProperty( VMContext* ctx, const String& prop ) const;


   ExprParentship* getParentship();

   static void applyInitExpr( VMContext* ctx, FalconClass* cls, FalconInstance* inst );

private:
   class Private;
   Private* _p;

   ExprParentship* m_parentship;
   SynFunc* m_constructor;

   bool m_shouldMark;
   bool m_hasInitExpr;
   bool m_hasInit;
   int m_missingParents;
   bool m_bPureFalcon;
   
   bool m_bConstructed;

   void initInstance( FalconInstance* inst ) const;
   void internal_callprop( VMContext* ctx, void* instance, FalconClass::Property& prop, int32 pCount ) const;

   // This is used to initialize the init expressions.
   class FALCON_DYN_CLASS PStepInitExpr: public PStep
   {
   public:
      PStepInitExpr( FalconClass* o );
      static void apply_( const PStep*, VMContext* );
      virtual void describeTo( String&, int depth=0 ) const;
   private:
      FalconClass* m_owner;
   };
   
   PStepInitExpr m_initExprStep;
};

}

#endif /* _FALCON_FALCONCLASS_H_ */

/* end of falconclass.h */
