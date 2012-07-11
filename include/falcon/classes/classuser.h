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
 
 Almost none of the classes in the framework is "complete" as is, and require
 specialization through subclassing.
 
 \section user_class_prop_meth Properties and Methods
 
 TODO
 
 \section user_class_prop_refl Reflective properties
 
 TODO
 
 \section user_class_uc About UserCarrier (and when \b not to use it)
 
 TODO
 
 */
class FALCON_DYN_CLASS ClassUser: public Class
{
public:
   ClassUser( const String& name );
   ClassUser( const String& name, int type );
   virtual ~ClassUser();
   
   /** Adds a property. 
    \param prop The propety (or method) to be added.
    
    Notice that this method is usually invoked directly by the property at creation.
    The added property is not destroyed by this class because it is considered
    owned by the subclass declaring them. They should be destroyed when the 
    subclass is disposed.
    */
   void add( Property* prop );
   
   /** Adds a parent class.
    This method allow to declare Class (most commonly another UserClass) as
    parent of this class.
    
    The parent classes will be used to determine automatically parentship
    when Class::derivedFrom is invoked, and will geneate an hidden read-only
    property having the same name of the parent class. The property is not
    returned in the property enumerations, and hasProperty returns false if
    invoked with the name of a parent class, but when invoked, an object having
    the desired parent class and the same user data as this class is returned.
    
    This is the common parent class access protocol in the standard class-based
    OOP model of Falcon, and is isomorphic with FalconClass behavior (classes
    declared at source file level or similarly built).
    
    This means that the parent classes must all understand and use the same
    user data associated with this child class. If this is not the case, e.g.
    if this class is theoretically a subclass of something, but the inner data
    the parents use is incompatible at binary level, then it is necessary to
    manually override Class::derivedFrom() and Class::getProperty to generate
    a proper instance for the Falcon engine.
    
    A safe way to use this automated feature is that of deriving the data
    this class presents to Falcon from the same UserData used by the parent(s).
    
    If the data type known by the parents is not compatible with the data type
    handled by this subclass, it is necessary just to override Class::getParentData()
    to offer a parent-specific data to the outer world. Notice that, in this case,
    it might be necessary to override also Class::gcMark() to correctly mark
    all the elements the parent classes need to mark.
    
    \note The parent classes added through this method are \b not invoked during
    object creation. 
    
    \note Subclasses providing a different inheritance scheme must override
    Class::getParent, Class::getParentData, Class::isDerivedFrom and Class::getProperty
    */
   void addParent( Class* cls );
   
   /** Overridden to return one of the parents declared through addParent(). 
    \param name The name of a parent.
    \return A parent added in addParent if the name is known, 0 otherwise.
    */
   virtual Class* getParent( const String& name ) const;
   
   /** Overridden to return exactly the class data.
    
    \param parent A parent class.
    \param data The data handled by this class.
    \return data if parent is this class or any parent declared by addParent,
      0 otherwise.
    
    This method is invoked by getProperty when the name of a parent class
    is requested as a property. 
    
    This method should be overridden by subclasses if some of the parents cannot
    handle the same data type used by this class.
    */
   virtual void* getParentData( Class* parent, void* data ) const;
   
   /** Overridden to return true if the required class is added through addParent. 
    \param cls A parent class.
    \return true if the parent is known, false otherwise.
    
    */
   virtual bool isDerivedFrom( const Class* cls ) const;
   
   /** Overridden to mark this class and all the parents through addParent. 
    \param mark A GC mark indicator.    
    */
   virtual void gcMark( uint32 mark );
   
   /** Number of properties carried (to be cached).
    
    Carried properties are those properties that require separate caching,
    usually but not necessarily performed automatically or semi-automatically 
    through UserCarrier.
    */
   uint32 carriedProps() const { return m_carriedProps; }
   
   //==================================================================
   // Things to be overridden if you don't want to use UserCarrier.
   //
   
   /** Override base Class::dispose.
    \param instance the Instance to be disposed.
    
    \note This class supposes that the instance is derived from UserCarrier.
    if this is not the case, override this method to use your data.
    
    \note parent classes declared with addParent() are not invoked to clear
    their data. In case of need, it is necessary to derive this method to
    properly handle parent class disposal.
    */
   virtual void dispose( void* instance ) const;
   
   /** Override base Class::clone.
    \param instance the Instance to be disposed.
    \return a new instance of this item.
    
    \note This class supposes that the instance is derived from UserCarrier.
    if this is not the case, override this method to use your data.
    
    \note parent classes declared with addParent() are not invoked to clone
    their data. In case of need, it is necessary to derive this method to
    properly handle parent class cloning.
    */
   virtual void* clone( void* insatnce ) const;

   /** Override base Class::gcMark.
    \param instance the Instance to be marked.
    \param mark The new mark.
    
    \note This class supposes that the instance is derived from UserCarrier.
    if this is not the case, override this method to use your data.
    
    \note parent classes declared with addParent() are not invoked to mark
    their data. In case of need, it is necessary to derive this method to
    properly handle parent class marking.
    */
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   
   /** Override base Class::gcCheck.
    \param instance the Instance to be checked.
    \return True if this item should be destroyed.
    
    \note This class supposes that the instance is derived from UserCarrier.
    if this is not the case, override this method to use your data.
    */
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
   

   //=====================================================================
   // End of things to be overridden if you don't want to use UserCarrier.
   //
   
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;   
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   
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
