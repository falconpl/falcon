/*
   FALCON - The Falcon Programming Language.
   FILE: parser/nonterminal.cpp

   Token representing a non-terminal grammar symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Apr 2011 12:54:26 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#define SRC "engine/parser/nonterminal.cpp"

#include <falcon/parser/nonterminal.h>
#include <falcon/parser/parser.h>
#include <falcon/trace.h>

#include <deque>
#include <set>


namespace Falcon {
namespace Parsing {

//==========================================================
// Helper classes
//

class NonTerminal::Private
{
public:
   typedef std::deque<Token*> TokenList;
   TokenList m_subTokens;

   Private() {}
   ~Private() {
      TokenList::iterator titer = m_subTokens.begin();
      while( titer != m_subTokens.end() )
      {
         Token* ti = *titer;
         if ( ti->isNT() )
         {
            NonTerminal* nt = static_cast<NonTerminal*>(ti);
            if( nt->isDynamic() )
            {
               delete nt;
            }
         }
         ++titer;
      }
   }

};

//=======================================================
// Main nonterminal class
//

NonTerminal::NonTerminal(const String& name,  bool bRightAssoc ):
   Token(name)
{
   m_eh = 0;
   m_handler = 0;
   m_currentSubNT = 0;

   m_bRightAssoc = bRightAssoc;
   m_isDynamic = false;
   m_bNonTerminal = true;
   m_bIsRecursive = false;
   _p = new Private;
}


NonTerminal::NonTerminal():
   Token("")
{
   m_eh = 0;
   m_handler = 0;
   m_currentSubNT = 0;

   m_isDynamic = false;
   m_bNonTerminal = true;
   m_bIsRecursive = false;
   _p = new Private;
}

NonTerminal::~NonTerminal()
{
   delete m_currentSubNT;
   delete _p;
}


int NonTerminal::arity() const
{
   return _p->m_subTokens.size();
}


Token* NonTerminal::term( int n ) const
{
   if( n < (int) _p->m_subTokens.size() )
   {
      return _p->m_subTokens[n];
   }
   return 0;
}

void NonTerminal::term( int n, Token* t )
{
   if( n < (int) _p->m_subTokens.size() )
   {
      if( t->isNT() )
      {
         m_bSimple = false;
      }
      else {
         if ( m_prio == 0 || t->prio() < m_prio )
         {
            m_prio = t->prio();
         }
      }
      _p->m_subTokens[n] = t;
   }
}


void NonTerminal::addTerm( Token* t )
{
   if(! _p->m_subTokens.empty() )
   {
      m_bSimple = false;
   }

   if( t->isNT() )
   {
      m_bSimple = false;
   }
   else {
      if ( m_prio == 0 || t->prio() < m_prio )
      {
         m_prio = t->prio();
      }
   }
   _p->m_subTokens.push_back(t);
}

void NonTerminal::render( TextWriter& tw ) const
{
   std::set<NonTerminal*> emptySet;
   subRender(tw, &emptySet);
}


void NonTerminal::subRender( TextWriter& tw, void* v ) const
{
   std::set<const NonTerminal*>& parentSet = *static_cast< std::set<const NonTerminal*>* >(v);
   parentSet.insert(this);

   tw.write( name() );
   tw.write( ":-\n" );

   Private::TokenList::const_iterator ti = _p->m_subTokens.begin();
   bool bFirst = true;
   std::set<const NonTerminal*> subtok;
   while(ti != _p->m_subTokens.end())
   {
      Token* t = *ti;
      tw.write("   ");
      if( bFirst )
      {
         bFirst = false;
         tw.write("  ");
      }
      else
      {
         tw.write("| ");
      }

      if (!t->name().empty())
      {
         String temp = "/* " +t->name() + " */ ";
         tw.write( temp );
         for( int i = (int) temp.length(); i < 20; ++i )
         {
            tw.write(" ");
         }
      }

      for( int i = 0; i < t->arity(); ++i )
      {
         Token* subt = t->term(i);
         if( i != 0 )
         {
            tw.write(", ");
         }

         tw.write(subt->name());
         if( subt->isNT() )
         {
            NonTerminal* nt = static_cast<NonTerminal*>(subt);
            if( parentSet.find(nt) == parentSet.end() && nt != this )
            {
               subtok.insert(nt);
            }
         }
      }

      tw.write("\n");

      ++ti;
   }
   tw.write("   ;\n");

   //now work on the sub parts.
   std::set<const NonTerminal*>::const_iterator subiter = subtok.begin();
   while( subiter != subtok.end() )
   {
      const NonTerminal* tok = *subiter;
      tok->subRender( tw, &parentSet );
      parentSet.insert(tok);
      ++subiter;
   }
}

NonTerminal& NonTerminal::sr(NonTerminal& nt)
{
   if( nt.m_currentSubNT != 0)
   {
      throw BuildError(nt, "Creating a sub-terminal, but the old one is still open");
   }

   nt.m_currentSubNT = new NonTerminal;
   nt.m_currentSubNT->setDynamic();
   return nt;
}

NonTerminal& NonTerminal::nr(NonTerminal& nt)
{
   if( nt.m_currentSubNT != 0)
   {
      nt._p->m_subTokens.push_back(nt.m_currentSubNT);
   }

   nt.m_currentSubNT = new NonTerminal;
   nt.m_currentSubNT->setDynamic();
   return nt;
}

NonTerminal& NonTerminal::endr(NonTerminal& nt)
{
   if( nt.m_currentSubNT == 0)
   {
      throw BuildError(nt, "Closing sub-terminal, but it was not opened");
   }

   nt._p->m_subTokens.push_back(nt.m_currentSubNT);
   nt.m_currentSubNT = 0;
   return nt;

}


NonTerminal& NonTerminal::addSubTerminal(Token& token)
{
   if( m_currentSubNT == 0)
   {
      throw BuildError(*this, "Adding sub-terminal but the subterminal is not open");
   }

   if( &token == this )
   {
      m_bIsRecursive = true;
   }
   m_currentSubNT->addTerm(&token);
   return *this;
}


NonTerminal& NonTerminal::addSubHandler(Handler hr)
{
   if( m_currentSubNT == 0)
   {
      throw BuildError(*this, "Adding sub-terminal but the subterminal is not open");
   }

   m_currentSubNT->setHandler(hr);
   return *this;
}


NonTerminal& NonTerminal::addSubName(const String& name)
{
   if( m_currentSubNT == 0)
   {
      this->name( name );
   }
   else
   {
      if( m_currentSubNT->name().empty() )
      {
         m_currentSubNT->name(name);
      }
      else
      {
         _p->m_subTokens.push_back(m_currentSubNT);
         m_currentSubNT = new NonTerminal(name);
         m_currentSubNT->setDynamic();
      }
   }

   return *this;
}

NonTerminal::BuildError::BuildError(const NonTerminal& src, const String& descr)
{
   m_descr = descr;
   m_descr.A(" in ").A(src.name()).A(":").N( (int32)(src._p->m_subTokens.size()+1) );
}

}
}

/* end of parser/nonterminal.cpp */
