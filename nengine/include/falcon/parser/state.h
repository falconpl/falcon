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

namespace Falcon {
namespace Parser {

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
   /** Support for variable parameter constructor idiom.
    To create a rule:
    State s = State::Maker("name").n( NonTerminal1 ).n( NonTerminal2 ).n( t_EOL ) )....;
    */
   class Maker
   {
      friend class State;

      inline Maker( const String& name );
      ~Maker();

      /** Adds a term or rule to this rule. */
      Maker& n( NonTerminal& t );

   private:
      const String& m_name;

      // inner tokens.
      mutable State::Private* _p;
   };

   State( const String& name );
   State( const Maker& maker );
   
   virtual ~State();

   const String& name() const {return m_name;}

   State& n( NonTerminal& e );

   void process( Parser& parser );
   
private:
   Private* _p;
   String m_name;
};

}
}

#endif	/* _FALCON_PARSER_STATE_H_ */

/* end of parser/state.h */

