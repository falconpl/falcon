/**
 *  \file gtk_HPaned.cpp
 */

#include "gtk_HPaned.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void HPaned::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_HPaned = mod->addClass( "GtkHPaned", &HPaned::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkPaned" ) );
    c_HPaned->getClassDef()->addInheritance( in );

    c_HPaned->getClassDef()->factory( &HPaned::factory );
}


HPaned::HPaned( const Falcon::CoreClass* gen, const GtkHPaned* paned )
    :
    Gtk::CoreGObject( gen, (GObject*) paned )
{}

Falcon::CoreObject* HPaned::factory( const Falcon::CoreClass* gen, void* paned, bool )
{
    return new HPaned( gen, (GtkHPaned*) paned );
}


/*#
    @class GtkHPaned
    @brief A container with two panes arranged horizontally

    The HPaned widget is a container widget with two children arranged horizontally.
    The division between the two panes is adjustable by the user by dragging a handle.
    See GtkPaned for details.
 */
FALCON_FUNC HPaned::init( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* wdt = gtk_hpaned_new();
    self->setObject( (GObject*) wdt );
}


} // Gtk
} // Falcon
