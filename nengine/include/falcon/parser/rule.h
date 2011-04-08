/*
   FALCON - The Falcon Programming Language.
   FILE: parser/token.h

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Apr 2011 17:16:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_RULE_H_
#define	_FALCON_PARSER_RULE_H_

#include <falcon/setup.h>

namespace Falcon {
namespace Parser {

/** Non-terminal parser rule.

 Rules are a set of the following things:
 - A sequence of tokens that MUST be matched (without possible alternatives).
 - A production '
 */
class FALCON_DYN_CLASS Rule: public Token
{
public:
   /** Support for variable parameter constructor idiom.
    To create a rule:
    Rule r = Rule::Maker("name").r( tok1 ).r( rule2.id() ).r( EOL::ID() ) )....;
    */
   class Maker
   {
      friend class Rule;

      inline Maker( const String& name ):
         m_name(name)
      {}

      /** Adds a term or rule to this rule. */
      Maker& r( uint32 id );

      void* _p;
   };

   Rule( const String& name );
   /** Initializer for variable parameter idiom. */
   Rule( const Maker& m );
   
   virtual ~Rule();

   /** Detach a deep value associated with this rule. */
   virtual void detachValue();

private:
   // Private inner data.
   void* _p;
};

}
}

#endif	/* _FALCON_PARSER_TOKEN_H_ */

/* end of parser/token.h */
