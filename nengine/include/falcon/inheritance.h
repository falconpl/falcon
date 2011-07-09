/*
   FALCON - The Falcon Programming Language.
   FILE: inheritance.h

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 13:35:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_INHERITANCE_H_
#define _FALCON_INHERITANCE_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

class Class;
class Expression;

/** Structure holding information about inheritance in a class.
  
 */
class FALCON_DYN_CLASS Inheritance
{
public:
   /** Creates the inheritance instance.
    \param name The logical name of the parent class.

    The inheritance name is the name of the class as it's written
    after the "from" clause in the inheritance declarartion.

    It may include namespaces and/or remote module names, and might
    not corespond with the name of the parent class as it is declared
    in the source module.

    */
   Inheritance( const String& name );
   ~Inheritance();

   /** The name of the class that we're searching.
    This includes full path/namespace.

    The inheritance name is the name of the class as it's written
    after the "from" clause in the inheritance declarartion.

    It may include namespaces and/or remote module names, and might
    not corespond with the name of the parent class as it is declared
    in the source module.
    */
   const String& className() const { return m_name; }

   /** The parent class.
    \return the Parent class, when resolved, or 0 if still not available.
    */
   Class* parent() const { return m_parent; }

   /** Sets the parent actually reference by this inheritance.
    \param cls The class that the owner class derivates from.
    */
   void parent( Class* cls ) { m_parent = cls; }

   /** Adds a parameter declaration.
      \param expr The expression that must be evaluated to generate the paramter.

    Inheritance parameters are expressions that must be calculated at runtime
    before invoking the init method of the subclass.
    */
   void addParameter( Expression* expr );

   /** Describes this inheritance.
      \param target A string where to place the description of this class.
    */
   void describe( String& target ) const;

   /** Describes this inheritance entry.
    \return A description of this entry.
    */
   String describe() const { 
      String target;
      describe( target );
      return target;
   }

private:

   String m_name;
   Class* m_parent;
};

}

#endif /* _FALCON_INHERITANCE_H_ */

/* end of inheritance.h */
