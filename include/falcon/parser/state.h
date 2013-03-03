/*
   FALCON - The Falcon Programming Language.
   FILE: parser/state.h

   Token representing a literal grammar terminal.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Apr 2011 17:25:49 +0200
 
   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_STATE_H_
#define	_FALCON_PARSER_STATE_H_

#include <falcon/setup.h>
#include <falcon/string.h>

#include "nonterminal.h"

namespace Falcon {
namespace Parsing {

class NonTerminal;
class Parser;

/** Grammar state.

 A state represents a set of non terminal tokens (each being a set of rules)
 which can be applied by the parser to parse an input.

 */
class FALCON_DYN_CLASS State
{
private:
   class Private;

public:
   State();
   State( const String& name );
   
   virtual ~State();

   const String& name() const {return m_name;}

   State& n( NonTerminal& e );

   bool findPaths( Parser& parser );

   /** Adds a top-level rule to this state. */
   inline State& operator<<( NonTerminal &nt ) { return n(nt); }
   /** Sets the name of this state. */
   inline State& operator<<( const String &name ) { m_name = name; return *this; }
   
private:
   Private* _p;
   String m_name;
};

}
}

#endif	/* _FALCON_PARSER_STATE_H_ */

/* end of parser/state.h */

