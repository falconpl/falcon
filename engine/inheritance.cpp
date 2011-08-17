/*
   FALCON - The Falcon Programming Language.
   FILE: inheritance.cpp

   Structure holding information about inheritance in a class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 13:35:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/inheritance.h>
#include <falcon/string.h>
#include <falcon/pcode.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>

#include <vector>


namespace Falcon
{

class Inheritance::Private
{
public:
   typedef std::vector<Expression*> ParamList;
   ParamList m_params;
   // precompiled params.
   PCode m_cparams;

   Private() {}
   
   ~Private()
   {
      ParamList::iterator iter = m_params.begin();
      while( m_params.end() != iter )
      {
         delete *iter;
         ++iter;
      }
   }
};


Inheritance::Inheritance( const String& name, Class* parent, Class* owner ):
   _p( new Private ),
   m_name(name),
   m_parent( parent ),
   m_owner( owner )
{

}

Inheritance::~Inheritance()
{
   delete _p;
}


void Inheritance::parent( Class* cls )
{
   fassert( m_parent == 0 );
   
   m_parent = cls;
   if( m_owner != 0 )
   {
      m_owner->onInheritanceResolved( this );
   }
}


void Inheritance::addParameter( Expression* expr )
{
   _p->m_params.push_back( expr );
   expr->precompile( &_p->m_cparams );
}


size_t Inheritance::paramCount() const
{
   return _p->m_params.size();
}

Expression* Inheritance::param( size_t n ) const
{
   return _p->m_params[n];
}


void Inheritance::describe( String& target ) const
{
   Private::ParamList& params = _p->m_params;

   if( params.empty() )
   {
      target = m_name;
   }
   else
   {
      target = m_name + "(";

      String temp;
      Private::ParamList::iterator iter = params.begin();
      while( params.end() != iter )
      {
         if( temp.size() >  0 )
         {
            temp += ", ";
         }
         temp += (*iter)->describe();
         ++iter;
      }
      
      target += temp + ")";
   }
}


 PCode* Inheritance::compiledExpr() const
 {
    return &_p->m_cparams;
 }

}

/* end of inheritance.cpp */
