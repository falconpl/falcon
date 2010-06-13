/**
 *  \file gtk_TreeIter.cpp
 */

#include "gtk_TreeIter.hpp"

#undef MYSELF
#define MYSELF Gtk::TreeIter* self = Falcon::dyncast<Gtk::TreeIter*>( vm->self().asObjectSafe() )


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeIter::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeIter = mod->addClass( "%GtkTreeIter" );

    c_TreeIter->setWKS( true );
    c_TreeIter->getClassDef()->factory( &TreeIter::factory );

    mod->addClassProperty( c_TreeIter, "stamp" );

    mod->addClassMethod( c_TreeIter, "copy", &TreeIter::copy );
}


TreeIter::TreeIter( const Falcon::CoreClass* gen, const GtkTreeIter* iter )
    :
    Falcon::CoreObject( gen )
{
    if ( iter )
        m_iter = *iter;
    else
        memset( &m_iter, 0, sizeof( GtkTreeIter ) );
}


TreeIter::~TreeIter()
{
}


bool TreeIter::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    if ( s == "stamp" )
        it = m_iter.stamp;
    else
        return defaultProperty( s, it );
    return true;
}


bool TreeIter::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* TreeIter::factory( const Falcon::CoreClass* gen, void* iter, bool )
{
    return new TreeIter( gen, (GtkTreeIter*) iter );
}


/*#
    @class GtkTreeIter
    @brief The GtkTreeIter is the primary structure for accessing a structure (model).
    @prop stamp A unique stamp to catch invalid iterators.
 */


/*#
    @method copy GtkTreeIter
    @brief Creates a copy of the tree iter.
    @return a new GtkTreeIter
 */
FALCON_FUNC TreeIter::copy( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(),
                                   self->getTreeIter() ) );
}


} // Gtk
} // Falcon
