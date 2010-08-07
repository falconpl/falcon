#ifndef GTK_RADIOTOOLBUTTON_HPP
#define GTK_RADIOTOOLBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::RadioToolButton
 */
class RadioToolButton
    :
    public Gtk::CoreGObject
{
public:

    RadioToolButton( const Falcon::CoreClass*, const GtkRadioToolButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_from_stock( VMARG );

    //static FALCON_FUNC get_group( VMARG );

    //static FALCON_FUNC set_group( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_RADIOTOOLBUTTON_HPP
