/*
   FALCON - The Falcon Programming Language.
   FILE: varmap.h

   Map holding local and global variable tables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 29 Dec 2012 10:03:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_VARMAP_H_
#define _FALCON_VARMAP_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class String;
class Variable;

/** Map holding local and global variable tables.
 *
 * This map holds variables that can be defined in functions
 * and modules.
 *
 * It has support to find a variable name given its type and
 * ID.
 *
 */
class FALCON_DYN_CLASS VarMap
{
public:
   VarMap();
   VarMap( const VarMap& other );
   ~VarMap();

   /** Adds a parameter to this function.
    \param name The parameter to be added.
    \return the parameter ID assigned to this variable.
    */
   Variable* addParam( const String& name );

   /** Adds a local variable to this function.
    \param name The local to be added.
    \return local variable ID assigned to this variable.

    If the variable name was already given, Variable::UNDEF is returned instead.
    */
   Variable* addLocal( const String& name );

   /** Adds a closed variable to this function.
    \param name The local to be added.
    \return local variable ID assigned to this variable.

    If the variable name was already given, Variable::UNDEF is returned instead.
    */
   Variable* addClosed( const String& name );

   Variable* addGlobal( const String& name );

   Variable* addExtern( const String& name );

   Variable* find( const String& name ) const;

   const String& getParamName( uint32 id ) const;
   const String& getLoacalName( uint32 id ) const;
   const String& getClosedName( uint32 id ) const;

   const String& getGlobalName( uint32 id ) const;
   const String& getExternName( uint32 id ) const;

   uint32 paramCount() const;
   uint32 localCount() const;
   uint32 closedCount() const;
   uint32 globalCount() const;
   uint32 externCount() const;

   uint32 allLocalCount() const;

   class Enumerator {
   public:
      virtual void operator()( const String& name, const Variable& var ) = 0;
   };

   void enumerate( Enumerator& e );

   bool isEta() const { return m_bEta; }
   void setEta( bool e ) { m_bEta = e; }

   void store( DataWriter* dw ) const;
   void restore( DataReader* dr );

private:
   class Private;
   VarMap::Private* _p;
   bool m_bEta;
};

}

#endif

/* end of varmap.h */
