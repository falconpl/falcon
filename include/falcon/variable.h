/*
   FALCON - The Falcon Programming Language.
   FILE: variable.h

   Name -- textual identifier of a variable.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_VARIABLE_H
#define FALCON_VARIABLE_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/fassert.h>

namespace Falcon {

class VMContext;
class CallFrame;
class Item;


/** Definition of a falcon variable.
 *
 * A Falcon variable is an item reachable at a certain location.
 *
 * Falcon variables are divided into the following categories:
 *
 * - Global variables stored in the global area
 *    of the module where the code resides.
 * - Parameter variables allocated in the previous frame of the data stack
 *   that then become parameters for subsequent functions.
 * - Local names allocated in the current call frame of the data stack.
 * - Closed names allocated in the closure data of the
 *   current call frame of the data stack.
 * - Extern a global variable stored in another module.
 *
 * Variables can be declared as constant
 */
class FALCON_DYN_CLASS Variable
{
public:
   typedef enum {
      e_nt_local,
      e_nt_param,
      e_nt_global,
      e_nt_closed,
      e_nt_extern,
      e_nt_undefined
   } type_t;

   /** Value of the ID of undefined symbols. */
   const static uint32 undef=(uint32)-1;

   /** The default constructor creates a minimally configured undefined variable
   */
   inline Variable():
            m_id(undef),
            m_declaredAt( 0 ),
            m_type(e_nt_undefined),
            m_isConst( false ),
            m_isResolved( false )
   { }

   inline  Variable( type_t type, uint32 id=undef, int32 declaredAt = 0, bool isConst=false ):
      m_id(id),
      m_declaredAt( declaredAt ),
      m_type(type),
      m_isConst( isConst ),
      m_isResolved( false )
   {
   }

   /** Copies the other variable */
   Variable( const Variable& other ):
      m_id(other.m_id),
      m_type(other.m_type),
      m_isConst(other.m_isConst),
      m_isResolved( false )
   {}

   ~Variable() {
   }

   type_t type() const { return m_type; }
   void type( type_t t ) { m_type = t; }

   uint32 id() const { return m_id; }
   void id( uint32 id ) { m_id = id; }

   bool isConst() const { return m_isConst; }
   void setConst( bool c ) { m_isConst = c; }

   int32 declaredAt() const { return m_declaredAt; }
   void declaredAt( int32 d ) { m_declaredAt = d; }

   bool isGlobalOrExtern() const { return m_type == e_nt_extern || m_type == e_nt_global; }

   bool isResolved() const { return m_isResolved; }
   void isResolved( bool b) { m_isResolved = b; }

protected:
   uint32 m_id;
   int32 m_declaredAt;
   type_t m_type;
   bool m_isConst;
   bool m_isResolved;
};

}

#endif

/* end of variable.h */
