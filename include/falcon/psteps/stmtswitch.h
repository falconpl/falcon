/*
   FALCON - The Falcon Programming Language.
   FILE: stmtswitch.h

   Syntactic tree item definitions -- switch statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 02 May 2012 21:18:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SWITCH_H_
#define _FALCON_SWITCH_H_

#include <falcon/psteps/switchlike.h>

namespace Falcon {

class Symbol;
class Expression;
class VMContext;

/** Handler for the switch statement.

 A branch can be added under multiple selectors.
 */
class FALCON_DYN_CLASS StmtSwitch: public SwitchlikeStatement
{
public:
   /** Create the switch statement.
    \param line The line where this statement is declared in the source.
    \param chr The character at which this statement is declared in the source.

    */
   StmtSwitch( int32 line = 0, int32 chr = 0 );

   StmtSwitch( const StmtSwitch& other );
   virtual ~StmtSwitch();

   virtual void renderHeader( TextWriter* tw, int32 depth ) const;
   virtual StmtSwitch* clone() const { return new StmtSwitch(*this); }
   
private:
   class Private;
   StmtSwitch::Private* _p;

   static void apply_( const PStep*, VMContext* ctx );
};
   
}

#endif

/* end of stmtswitch.h */
