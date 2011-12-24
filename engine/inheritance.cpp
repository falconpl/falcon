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
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/module.h>

#include <falcon/symbol.h>
#include <falcon/falconclass.h>

#include <falcon/errors/codeerror.h>

#include <vector>


namespace Falcon
{

class Inheritance::Private
{
public:
   typedef std::vector<Expression*> ParamList;
   ParamList m_params;

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
   m_owner( owner ),
   m_requirer( name, this )
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
}


size_t Inheritance::paramCount() const
{
   return _p->m_params.size();
}

Expression* Inheritance::param( size_t n ) const
{
   return _p->m_params[n];
}


bool Inheritance::prepareOnContext( VMContext* ctx )
{
   Private::ParamList& params = _p->m_params;
   if( params.empty() )
   {
      return false;
   }
   
   Private::ParamList::iterator iter = params.begin();
   while( params.end() != iter )
   {
      ctx->pushCode( *iter ); 
      ++iter;
   }
   return true;
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


void Inheritance::IRequirement::onResolved(   
         const Module* source, const Symbol* srcSym, Module* tgt, Symbol* )
{
   const Item* value;
   
   if( (value = srcSym->value( 0 )) == 0 || ! value->isClass() )
   {
      // the symbol is not a class?   
      throw new CodeError( ErrorParam( e_inv_inherit ) 
         .module( source == 0 ? "<internal>" : source->uri() )
         .symbol( srcSym->name() )
         .line( m_owner->sourceRef().line())
         .chr( m_owner->sourceRef().chr())
         .origin(ErrorParam::e_orig_linker));
   }

   // Ok, we have a valid class.
   Class* newParent = static_cast<Class*>(value->asInst());
   m_owner->parent( newParent );
   Class* cls = m_owner->m_owner;
   // is the owner class a Falcon class?
   if( cls->isFalconClass() )
   {
      // then, see if we can link it.
      FalconClass* falcls = static_cast<FalconClass*>(cls);
      if( falcls->missingParents() == 0 && tgt != 0 )
      {
         // ok, the parent that has been found now was the last one.
         tgt->completeClass( falcls );
      }
   }
}
  
 
}

/* end of inheritance.cpp */
