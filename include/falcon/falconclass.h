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
         size_t id;
         Function* func;
         ExprInherit* inh;
         FalconState* state;
      } Value;

      typedef enum tag_proptype {
         t_prop,
         t_func,
         t_inh,
         t_state
      } Type;

      String m_name;
      Type m_type;
      Value m_value;

      Property( const Property& other );

      inline Property( const String& name, size_t value ):
         m_name( name ),
         m_type(t_prop),
         m_expr(0)
      {
         m_value.id = value;
      }

      Property( const String& name, size_t value, Expression* expr );

      Property( Function* value );
      Property( ExprInherit* value );
      Property( FalconState* value );


      ~Property();

      /** Returns associated expression (if this property holds an expression).
       \return a valid PCode or 0 if this is not an expression.
       */
      Expression* expression() const { return m_expr; }

      Property* clone() const { return new Property(*this); }
   private:
      Expression* m_expr;
   };

   FalconClass( const String& name );
   virtual ~FalconClass();

   /** Returns the name of the falcon class.
    \return the name of this class.
    */
   const String& fc_name() const { return m_fc_name; }

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
   bool addProperty( const String& name, const Item& initValue );

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
   bool addMethod( Function* mth );

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
   virtual void* getParentData( Class* parent, void* data ) const;

   //====================================================================
   // Overrides from Class
   //

   virtual Class* getParent( const String& name ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   //=========================================================
   // Class management
   //

   /** Marks the items in this class (if considered necessary).
    \param mark The mark that is used by GC.

    This is used to mark the default values of the class, if
    there is one or more deep item in the class.
    */
   virtual void gcMarkMyself( uint32 mark );

   virtual void gcMark( void* self, uint32 mark ) const;

   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;


   /** Creates the constructor or returns it if it's already here. */
   SynFunc* makeConstructor();

   /** Return the constructor of this class. */
   SynFunc* constructor() const { return m_constructor; }

   /** Create the class structure compiling it from its parents.
    \param bHiddenParents If true, we have some parents that might not
    be in our parent list.
    \return false if there is still some unknown parent,
            true if the construction process is complete.

    This is called by the VM after all the missing parents have been
    found, provided that all the parents are declared as FalconClass subclasses.

    If some of the parents are not FalconClass, the engine must generate an
    hyperclass out of this falcon class.

    This method may destroy the constructor (or return succesfully without
    actually creating it) if it detects that we have no parents and we don't
    have nothing to be initialized. If bHiddenParents If true, we have some
    parents that might not be in our parent list. This means that the symbol
    talbe of our constructor might be in use; so it means that the constructor
    must not be disposed of if already existing.
    */
   bool construct( bool bHiddenParents = false );

   /** Return the count of currently unknown parents.
    \return The count of unresolved inheritances.
    */
   int missingParents() const { return m_missingParents; }

   /** Check if all the declared inheritances are pure falcon classes.
    \return True if this class can be generated as a Falcon Class.

    To construct a Fa

    */
   bool isPureFalcon() const { return m_bPureFalcon; }

   /** Construct this class as an hyperclass.
    \return an HyperClass representing this FalconClass.
    \note after this call, the returned hyperclass is the sole owner of this
          FalconClass instance. Every other reference to this FalconClass
          should be abandoned.
    */
   HyperClass* hyperConstruct();

   /** Called back by an inheritance when it gets resolved.
    \param inh The inheritance that has been just resolved.

    When a foreign inheritance of a class gets resolved during the link
    phase, the parent need to know about this fact to prepare itself.

    The inheritance determines if the owner is a falcon class, and in case
    it is, it calls back this method.

    \note The class hierarcy itself doesn't need to know about resovled
    inheritances, as normally the classes can be formed only when all their
    components are known. Third party user-classes must pre-load the
    required components and pre-resolve their dependencies. Prototypes
    are created at runtime when all the dependencies are known, and hyperclasses
    follow the rules of third party classes. In short, the only class type that
    supports forward definition of parentship is FalconClass.

    On a FalconClass instance, this determines if the class is a pure
    FalconClass or needs to be transformed in an HyperClass.
    */
   virtual void onInheritanceResolved( ExprInherit* inh );

   //=========================================================
   // Operators.
   //

   virtual void op_init( VMContext* ctx, int32 pcount ) const;
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;

private:
   class Private;
   Private* _p;

   ExprParentship* m_parentship;
   String m_fc_name;
   SynFunc* m_constructor;

   Function** m_overrides;

   bool m_shouldMark;
   bool m_hasInitExpr;
   bool m_hasInit;
   int m_missingParents;
   bool m_bPureFalcon;

   // This is used to initialize the init expressions.
   class PStepInitExpr: public Statement
   {
   public:
      PStepInitExpr( FalconClass* o );
      static void apply_( const PStep*, VMContext* );
      virtual PStepInitExpr* clone() const { return 0; }
   private:
      FalconClass* m_owner;
   };

   // This is used to invoke
   class PStepInit: public Statement
   {
   public:
      PStepInit( FalconClass* o );
      static void apply_( const PStep*, VMContext* );
      virtual PStepInitExpr* clone() const { return 0; }

   private:
      FalconClass* m_owner;
   };
   friend class PStepInitExpr;
};

}

#endif /* _FALCON_FALCONCLASS_H_ */

/* end of falconclass.h */
