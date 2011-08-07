/*
   FALCON - The Falcon Programming Language.
   FILE: classuser.h

   A class with some automation to help reflect foreign code.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 07 Aug 2011 12:11:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSUSER_H_
#define _FALCON_CLASSUSER_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/class.h>

namespace Falcon {

class Property;

/** Instance of a class with some user-oriented automation.
 
 The base Class model is extremely flexible, but it requires some repetitive
 code to be written in case of integration with foreign and complex code.
 
 This class leverages on the base Class flexibility (and performance) to create 
 some structured access rules that are easier to write and maintain.
 
 \note Actually, the downgrade in performance is always marginal and can even
 be reversed, proving faster than an hand-made ad-hoc solution, in case of 
 complex objects are managed through it.
 
 The ClassUser is still a base class that must be re-implemented by the final 
 users. This also allows to cherry-pick desirable features and drop those that
 are considered unworth or unnecessary by reimplementing some virtual functions.
 
 The ClassUser leave undefined the Class::serialize and Class::deserialize
 abstract methods, and adds a new abstract method called createInstance. It's
 possible to stub all this methods as doing-nothing if they are not needed.
 
 \section user_class_components Components of a user class.
 
 To authomatize the integration of foreign code, the ClassUser framework
 provides three devices whose usage can be moduleated depending on the final
 needs.
 
 - Property: An abstraction of a property known to the class. The final user
   should create properties which provide a way to access to the instance data.
 - Method: An abstraction of a function that responds to a certain property
   name known in the class.
 - UserCarrier: A class containing some minimal helpful devices to integrate
   external data. If the foreign instance to be wrapped is simple enough, or if
   it is already designed to be shared with the Falcon engine, it
   is superfluous and it can be dropped, but it offers facilities that are
   commonly required by any non-elementary integration of foreign unaware code,
   as deep GC marking, item reflection, cloning and destruction.
 
 Almost none of the class in the framework is "complete" as is, and require
 specialization through subclassing.
 
 \section user_class_prop_meth Properties and Methods
 
 TODO
 
 \section user_class_prop_refl Reflective properties
 
 TODO
 
 \section user_class_uc About UserCarrier (and when \b not to use it)
 
 TODO
 
 */
class ClassUser: public Class
{
public:
   ClassUser( const String& name );
   virtual ~ClassUser();
   
   /** Adds a property. 
    \param prop The propety (or method) to be added.
    
    Notice that this method is usually invoked directly by the property at creation.
    The added property is not destroyed by this class because it is considered
    owned by the subclass declaring them. They should be destroyed when the 
    subclass is disposed.
    */
   void add( Property* prop );
   
   /** Number of properties carried (to be cached).
    
    Carried properties are those properties that requrie separate caching,
    usually but not necessarily performed automatically or semi-automatically 
    through UserCarrier.
    */
   uint32 carriedProps() const { return m_carriedProps; }
   
   /** Instance creation helper.
    \param params An array of items containing the parameters passed by the VM.
    \param pcount Number of parameters in the params array.
    
    This method is called back by the op_create when creating an object
    of this class. Actually, the op_create is just a wrapper adround this class,
    so if you want to use directly the op_create method, you can just stub out
    this abstract virtual call and write the needed code in op_create.
    
    However, notice that op_create needs to:
    - Get the parameter array
    - Account the created item in the GC, and create a poper UserData item.
    - properly uroll the stack and place the created item at proper location on 
      success.
    
    Be sure to know the op_create requirements if you want to use it directly.
    
    \note This method \b must return a valid instance. In case this is not
    possible, it \b must throw a relevant Error derived exception pointer.
    */
   virtual void* createInstance( Item* params, int pcount ) const = 0;
   
   //==================================================================
   // Things to be overridden if you don't want to use UserCarrier.
   //
   
   /** Override base Class::dispose.
    \param instance the Instance to be disposed.
    
    \note This class supposes that the instance is derived from UserCarrier.
    if this is not the case, override this method to use your data.
    */
   virtual void dispose( void* instance ) const;
   
   /** Override base Class::clone.
    \param instance the Instance to be disposed.
    \return a new instance of this item.
    
    \note This class supposes that the instance is derived from UserCarrier.
    if this is not the case, override this method to use your data.
    */
   virtual void* clone( void* insatnce ) const;

   /** Override base Class::gcMark.
    \param instance the Instance to be marked.
    \param mark The new mark.
    
    \note This class supposes that the instance is derived from UserCarrier.
    if this is not the case, override this method to use your data.
    */
   virtual void gcMark( void* instance, uint32 mark ) const;
   
   /** Override base Class::gcCheck.
    \param instance the Instance to be checked.
    \return True if this item should be destroyed.
    
    \note This class supposes that the instance is derived from UserCarrier.
    if this is not the case, override this method to use your data.
    */
   virtual bool gcCheck( void* instance, uint32 mark ) const;
   

   //=====================================================================
   // End of things to be overridden if you don't want to use UserCarrier.
   //
   
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;   
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   
   virtual void op_create( VMContext* ctx, int32 pcount ) const;
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop ) const;

   
private:
   class Private;
   Private* _p;
   
   uint32 m_carriedProps;
};

}

#endif	/* _FALCON_USERCARRIER_H_ */

/* end of classuser.h */
