#ifndef GTK_TOGGLEACTION_HPP
#define GTK_TOGGLEACTION_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ToggleAction
 */
class ToggleAction
    :
    public Gtk::CoreGObject
{
public:

    ToggleAction( const Falcon::CoreClass*, const GtkToggleAction* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_toggled( VMARG );

    static void on_toggled( GtkToggleAction*, gpointer );

    static FALCON_FUNC toggled( VMARG );

    static FALCON_FUNC set_active( VMARG );

    static FALCON_FUNC get_active( VMARG );

    static FALCON_FUNC set_draw_as_radio( VMARG );

    static FALCON_FUNC get_draw_as_radio( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TOGGLEACTION_HPP
