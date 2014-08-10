/*
   FALCON - The Falcon Programming Language.
   FILE: mantra.h

   Basic "utterance" of the Falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 25 Feb 2012 21:38:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)
*/
  
#ifndef FALCON_MANTRA_H
#define FALCON_MANTRA_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/sourceref.h>
#include <falcon/attributemap.h>
#include <falcon/delegatemap.h>

namespace Falcon 
{

class Class;
class Module;
class TreeStep;

/** Basic "utterance" of the Falcon engine.
 
 A mantra is a basic entity that is bound to have a specific effect on the
 Falcon engine.
 
 They are mainly divided into two categories:
 
 - Function: collection of executable code that the engine can run.
 - Class: Entity describing an open set of data, or in other words, entity
 acting as a data handler for the engine.
 
 Mantras can be simply "given" in the engine or provided by modules. They
 
 */
class FALCON_DYN_CLASS Mantra
{
public:
   typedef enum {
      e_c_none,
      e_c_function,
      e_c_pseudofunction,
      e_c_synfunction,
      
      e_c_class,
      e_c_falconclass,
      e_c_hyperclass,
      e_c_metaclass,
      e_c_object
   }
   t_category;
   
   /** Class representing the coordinates of a mantra in the system.
    A mantra must have a name.
    It might have a module logical name if it's crated in a logical module.
    It might have a module physical name (URI) if it's created from outside.
    
    Searching a mantra by logical module name, if given, takes precedence,
    however if a logical name and a phyiscal name are given, the physical name
    can be used if the module is not known under its logical name. It is said
    that a module URI "subsides" its logical name.
    
    If neither a logical name nor a URI are given, the entity is to be found
    in the engine itself.
    
    Giving a category can be used to restrict the search: 
    - e_c_none Indicates that  any mantra at the given location, if found, is valid
    - e_c_function is valid for functions, pseudofunctions and synfunctions.
    - e_c_class is valid for any class.
    */
   class Location 
   {
   public:
      String m_name;
      String m_moduleName;
      String m_moduleURI;
      t_category m_category;
      
      Location( const String& name ):
         m_name( name ),
         m_category( e_c_none )
      {}
      
      Location( const String& name, const String& modName ):
         m_name( name ),
         m_moduleName( modName ),
         m_category( e_c_none )
      {}
      
      Location( const String& name, const String& modName, const String& modURI ):
         m_name( name ),
         m_moduleName( modName ),
         m_moduleURI( modURI ),
         m_category( e_c_none )
      {}
      
      Location( const String& name, const String& modName, const String& modURI, t_category cat ):
         m_name( name ),
         m_moduleName( modName ),
         m_moduleURI( modURI ),
         m_category( cat )
      {}
      
      ~Location() {}
   };
   
   /** Creates an empty mantra.
      Used mainly during deserialization.
      A mantra without a name can be created, but it can't be used.
    */
   Mantra();   
   
   Mantra( const String& name, int32 line=0, int32 chr=0 );
   Mantra( const String& name, Module* module, int32 line=0, int32 chr=0 );
   
   virtual String fullName() const;

   virtual ~Mantra() {}
   
   t_category category() const { return m_category; }
   
   void name( const String& n ) { m_name = n; }
   
   const String& name() const { return m_name; }

   /** Returns the handler for this mantra.
    A handler is a class, which is a special kind of mantra.
    
    The base class returns a very simple ClassMantra class, which
    handles an abstract mantra.
    */
   virtual const Class* handler() const;
   
    /** Sets the module of this mantra.
    Mainly, this information is used for debugging (i.e. to know where a function
    is declared).
    */
   void module( Module* owner ) { m_module = owner; }

   /** Return the module where this function is allocated.
   */
   Module* module() const { return m_module; }
   
   SourceRef& sr() { return m_sr; }
   const SourceRef& sr() const { return m_sr; }
   
   /** Returns the source line where this function was declared.
    To be used in conjunction with module() to pinpoint the location of a function.
    */
   int32 declaredAt() const { return m_sr.line(); }

   void declaredAt( int32 line ) { m_sr.line(line); }

   /** Describe the location of the mantra.
    Format is basically:
    \code
      logical.module.name[(line[:char])]|<internal>:name [module URI]
    \endcode
    */
   virtual void locateTo( String& target ) const;
   
   String locate() const { String temp; locateTo( temp ); return temp; }
   
   /** GCMark this mantra.
      
    Virtual because some function subclasses having closed items
      may need to mark their own items.
    
    Used by handlers when this entity is being checked for vitality.
    */
   virtual void gcMark( uint32 mark );

   /** Checks if this mantra mark is up to date.
    Used by handlers when this entity is being checked for vitality.
    */
   inline bool gcCheck( uint32 mark ) { return m_mark >= mark; }
   
   /** Checks wether this mantra is compatible with a required category.
    \param cat A category.
    \return true if the mantra can be casted to the intended category,
            false otherwise.
   */
   bool isCompatibleWith( t_category cat ) const;

   const AttributeMap& attributes() const { return m_attributes; }
   AttributeMap& attributes() { return m_attributes; }

   bool addAttribute( const String& name, TreeStep* generator );

   virtual void render( TextWriter* tw, int32 depth ) const = 0;

   const DelegateMap& delegates() const { return m_delegates; }
   DelegateMap& delegates() { return m_delegates; }

protected:
   t_category m_category;
   String m_name;
   Module* m_module;
   SourceRef m_sr;
   AttributeMap m_attributes;
   DelegateMap m_delegates;
   
   uint32 m_mark;
};

}
#endif

/* end of mantra.h */
