/*
   FALCON - The Falcon Programming Language.
   FILE: parser/rule.h

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
#include <falcon/string.h>
#include <falcon/parser/matchtype.h>

namespace Falcon {
namespace Parsing {

class Parser;
class NonTerminal;
class Token;

/** Non-terminal parser rule.
 
 Rules identify a sequence of tokens that, if matched, cause them to be
 "applied".

 The rule application has the role to create part of the output, if this rule is a
 top-level rule, or to transform the tokens into something else if this is
 a middle-level rule. Actually, this distinction is not known at parser level,
 and it's the rule application that decides if it should generate some output
 or create a temporary new representation of its state.

 Although this may seem less structured than requesting the rules to conform
 to a pre-determined protocol, the Parser system offer a common interface to
 the rules, which makes easy to interact with it in a consistent way, with the
 same typing, or actually less, than you'd expect in a pure call-parameters-return-value
 parser-to-rule interaction.

 OTOH, you gain flexibility and control that is not usually available in more
 "structured" parser layouts.

 Due to the structure of the C++ language, creating more specific rules
 by subclassing the Rule class would be correct, but complex, error prone and
 inelegant. Instead, this Rule class represents a single sequence of tokens that,
 when matched, requires the application of a specific Apply functor (which,
 if appropriate, may even be shared across multiple rules).

 Not tokens nor Apply functors are owned by a rule. They must be statically
 stored elsewhere (or anyhow outlive all the Rule instances they are referred in).

 Notice that rules cannot have repetitions or alternatives; they just represent
 linear sets of tokens that have to be matched. A NonTerminal is a Token that can
 hold one or more alternative rules; once applied, a rule might substitute
 some tokens in the current Parser stack with a new NonTerminal token to which
 it belongs.

 */
class FALCON_DYN_CLASS Rule
{
public:
   /** Functor invoked when a rule is matched. */
   typedef void(*Apply)( const Rule& r, Parser& p );

   /** Direct constructor.
    \param name A symbolic name for this rule (useful in debug).
    \param app An Apply functor that will be invoked when the rule is matched.

    Use this to create a rule and fill tokens later.
    */
   Rule( const String& name, Apply app );

   Rule();
   
   virtual ~Rule();

   /** Adds a term to this rule. */
   Rule& t( Token& t );

   /** Equality operator used for variable parameter idiom.

    @code
       Rule r = Rule::Maker("name", R_Apply ).r( terminal1 ).r( NonTerminal2 ).r( t_EOL ) )....;
    @endcode
   */
   //Rule& operator =(const Maker&m );

   /** Accessor for rule name.
    */
   const String& name() const { return m_name; }   

   /** Checks if the rule is currently matching. */
   bool match( Parser& p, bool bIncremental ) const;

   /** Sets the non-terminal in which this rule is stored.
    \param nt The non-terminal symbol using this rule.

    This method is used by the non-terminal class when this rule
    is assigned to it.

    Setting this also checks for recursiveness of the rule.
    */
   void parent( NonTerminal& nt );
   const NonTerminal& parent() const { return *m_parent; }
   NonTerminal& parent() { return *m_parent; }

   /** Apply the rule.
    \param parser The parser that is currently applying this rule.
    
    This method cause a valid rule to be applied on the parser.
    */
   void apply( Parser& parser ) const;

   Rule& operator <<( Token& tok ) { return t(tok); }
   Rule& operator <<( const String& n ) { m_name = n; return *this; }
   Rule& operator <<( Apply func ) { m_apply = func; return *this; }

   /** Count of tokens in the rule.
    \return The number of tokens that are needed for this rule to match.
    */
   int arity() const;
   Token* getTokenAt( uint32 pos ) const;

   bool isGreedy() const { return m_bGreedy; }
   bool isRecursive() const { return m_bRecursive; }

   void setGreedy( bool mode ) { m_bGreedy = mode; } 
private:
   class Private;
   friend class Parser;
   
   String m_name;
   Apply m_apply;
   NonTerminal* m_parent;

   bool m_bGreedy;
   bool m_bRecursive;
   
   // Inner tokens
   Private* _p;
};

}
}

#endif	/* _FALCON_PARSER_RULE_H_ */

/* end of parser/rule.h */
