/**
 *  \file gtk_Progress.cpp
 */

#include "gtk_Progress.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Progress::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Progress = mod->addClass( "GtkProgress", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkWidget" ) );
    c_Progress->getClassDef()->addInheritance( in );

    c_Progress->getClassDef()->factory( &Progress::factory );
#if 0
    Gtk::MethodTab methods[] =
    {
    { "set_show_text",          &Progress::set_show_text },
    { "set_text_alignment",     &Progress::set_text_alignment },
    { "set_format_string",      &Progress::set_format_string },
    { "set_adjustment",         &Progress::set_adjustment },
    { "set_percentage",         &Progress::set_percentage },
    { "set_value",              &Progress::set_value },
    { "get_value",              &Progress::get_value },
    { "set_activity_mode",      &Progress::set_activity_mode },
    { "get_current_text",       &Progress::get_current_text },
    { "get_text_from_value",    &Progress::get_text_from_value },
    { "get_current_percentage", &Progress::get_current_percentage },
    { "get_percentage_from_value",&Progress::get_percentage_from_value },
    { "configure",              &Progress::configure },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Progress, meth->name, meth->cb );
#endif
}


Progress::Progress( const Falcon::CoreClass* gen, const GtkProgress* progress )
    :
    Gtk::CoreGObject( gen, (GObject*) progress )
{}


Falcon::CoreObject* Progress::factory( const Falcon::CoreClass* gen, void* progress, bool )
{
    return new Progress( gen, (GtkProgress*) progress );
}


/*#
    @class GtkProgress
    @brief Base class for GtkProgressBar

    A GtkProgress is the abstract base class used to derive a GtkProgressBar which
    provides a visual representation of the progress of a long running operation.
 */


#if 0
FALCON_FUNC Progress::set_show_text( VMARG );
FALCON_FUNC Progress::set_text_alignment( VMARG );
FALCON_FUNC Progress::set_format_string( VMARG );
FALCON_FUNC Progress::set_adjustment( VMARG );
FALCON_FUNC Progress::set_percentage( VMARG );
FALCON_FUNC Progress::set_value( VMARG );
FALCON_FUNC Progress::get_value( VMARG );
FALCON_FUNC Progress::set_activity_mode( VMARG );
FALCON_FUNC Progress::get_current_text( VMARG );
FALCON_FUNC Progress::get_text_from_value( VMARG );
FALCON_FUNC Progress::get_current_percentage( VMARG );
FALCON_FUNC Progress::get_percentage_from_value( VMARG );
FALCON_FUNC Progress::configure( VMARG );
#endif


} // Gtk
} // Falcon
