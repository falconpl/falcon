#ifndef GTK_TOGGLETOOLBUTTON_HPP
#define GTK_TOGGLETOOLBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ToggleToolButton
 */
class ToggleToolButton
    :
    public Gtk::CoreGObject
{
public:

    ToggleToolButton( const Falcon::CoreClass*, const GtkToggleToolButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_toggled( VMARG );

    static void on_toggled( GtkToggleToolButton*, gpointer );

    static FALCON_FUNC new_from_stock( VMARG );

    static FALCON_FUNC set_active( VMARG );

    static FALCON_FUNC get_active( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TOGGLETOOLBUTTON_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
