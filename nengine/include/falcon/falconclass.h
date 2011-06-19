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
#include <falcon/enumerator.h>

namespace Falcon
{

class FalconInstance;
class Inheritance;
class Item;
class Function;
class DataReader;
class DataWriter;
class FalconState;
class Expression;
class CoreClass;

/** Class defined by a Falcon script.

 This class implements a script definition of a class in a falcon script.

 This structure is adequate to hold properties, methods, states, an init block
 and class parentship as they are declared in a script through the "class"
 declaration.

 Of course, it is possible to synthethize a FalconClass in a C++ extension or
 module; also, a method can be any Falcon function, including external
 functions. It is also possible to dynamically create new classes, or alter
 the methods and states of existing ones.

 FalconClasses are handled by the engine through the CoreClass handler.

 This class has also the necessary data to create "instances". A FalconInstance
 is an ordered array of values that match the properties in the class, and
 is handled through a CoreInstance handler, which reads the properties in the
 FalconClass that is associated with the instance.

 \note Falcon classes can have either an empty parentship or can be derived from
 other FalconClass entities. Deriving a script class from a parent which is not
 a FalconClass itself (or from multiple parents, where one of the parents
 is not a FalconClass is implemented by creating an HyperClass instance.

 \note It is possible to access "static" methods and base classes from a live
 script.

 \note Despite the fact that modules created from source Falcon scripts will
 always create FalconClass instances, the VM may generate an HyperClass in
 their place if one of their parent classes is found to be a non-FalconClass
 instance during the link process, while resolving external symbols.
 
 */
class FALCON_DYN_CLASS FalconClass
{
public:


   class Property
   {
   public:
      typedef union tag_cnt
      {
         size_t id;
         Function* func;         
         Inheritance* inh;
         FalconState* state;
      } Value;

      typedef enum tag_proptype {
         t_prop,
         t_func,
         t_expr,
         t_inh,
         t_state
      } Type;

      Type m_type;
      Value m_value;

      Expression* m_expr;

      inline Property( size_t value ):
         m_type(t_prop),
         m_expr( 0 )
      {
         m_value.id = value;
      }

      inline Property( Function* value ):
         m_type(t_func),
         m_expr( 0 )
      {
         m_value.func = value;
      }

      inline Property( size_t value, Expression* expr ):
         m_type( t_expr ),
         m_expr( expr )
      {
         m_value.id = value;
      }

      inline Property( Inheritance* value ):
         m_type(t_inh),
         m_expr( 0 )
      {
         m_value.inh = value;
      }

      inline Property( FalconState* value ):
         m_type(t_state),
         m_expr( 0 )
      {
         m_value.state = value;
      }

      ~Property();
   };


   FalconClass( const String& name );
   virtual ~FalconClass();

   /** Returns the name of this class.
    \return the name of this class.
    */
   const String& name() const { return m_name; }

   /** Creates a new instance.
    \return A new instance created out of this class.
    The method will return 0 if the class is not completely resolved,
    i.e. if there is some parent that couldn't be found.
   */
   FalconInstance* createInstance() const;

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
    \param initExpr The expression that is used to initialize the proeprty.
    \return True if the property could be added, false if the name was already
    used.

    Properties added through this method will be initialized in the same order
    they have been added. They could theretically refer other properties in
    the "self" entities, so the initialization order is relevant.

    The init method, if present, is invoked after all the expressions have been
    evaluated.

    \note The expression is property of this class and will be destroyed when the
    class is destroyed.
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

   /** Adds an init handler.
      \param init The Function instance that will be used to initialize this class.
      \return true If the init was not given, false if another init was already set.
    */
   bool addInit( Function* init );

   /** Returns the init handler of this class, if any. */
   Function* init() const { return m_init; }

   /** Adds a parent and the parentship declaration.
    \param inh The inheritance declaration.
    \return true If the init was not given, false if another init was already set.
    The parentship delcaration carries also declaration for the parameters
    and the expressions needed to create the instances.

    The ownership of the inheritance records is held by this FalconClass.
    */
   bool addParent( Inheritance* inh );

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
   bool getMember( const String& name, Item& target ) const;

   /** Gets a member of this class -- as a definition.
    \param name The name of the member to be returned.
    \return A pointer to a property in this class or zero if not found.

    This method is mainly used by FalconInstance to get the member
    item in the class and then return a coherent data type to the VM
    that has accessed the item.
    */
   const Property* getMember( const String& name ) const;

   /** Marks the items in this class (if considered necessary).
    \param mark The mark that is used by GC.

    This is used to mark the default values of the class, if
    there is one or more deep item in the class.
    */
   void gcMark( uint32 mark ) const;

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

   /** Serialize this class to a writers. */
   void serialize( DataWriter* dw ) const;
   
   /** Deserialize this class from a reader.
    \param dr The reader
    This method is meant to be called.
    */
   void deserialize( DataReader* dr );


   /** Callback receiving all the properties in this class. */
   typedef Enumerator<String> PropertyEnumerator;

   /** List the members in this class having property semantics.
     @param cb A callback function receiving one property at a time.
    */
   void enumerateProperties( PropertyEnumerator& cb ) const;

   /** List all the properties in this class.
     @param cb A callback function receiving one property at a time.
    */
   void enumerateMembers( PropertyEnumerator& cb ) const;

   /** Check if this class has a given membeer.
    \param name The name of the property to be searched.
    \return true if the given property is present.
    */
   bool hasMember( const String& name ) const;

private:
   // used in deserialization
   FalconClass();

   class Private;
   Private* _p;
      
   String m_name;
   bool m_shouldMark;
   Function* m_init;

   friend class CoreClass;
};

}

#endif /* _FALCON_FALCONCLASS_H_ */

/* end of falconclass.h */
