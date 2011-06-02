/**
 *  \file gtk_Range.cpp
 */

#include "gtk_Range.hpp"

#include "gdk_Rectangle.hpp"

#include "gtk_Adjustment.hpp"
#include "gtk_Buildable.hpp"
#include "gtk_Orientable.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Range::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Range = mod->addClass( "GtkRange", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkWidget" ) );
    c_Range->getClassDef()->addInheritance( in );

    //c_Range->getClassDef()->factory( &Range::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_adjust_bounds",       &Range::signal_adjust_bounds },
    { "signal_change_value",        &Range::signal_change_value },
    { "signal_move_slider",         &Range::signal_move_slider },
    { "signal_value_changed",       &Range::signal_value_changed },
    { "get_fill_level",             &Range::get_fill_level },
    { "get_restrict_to_fill_level", &Range::get_restrict_to_fill_level },
    { "get_show_fill_level",        &Range::get_show_fill_level },
    { "set_fill_level",             &Range::set_fill_level },
    { "set_restrict_to_fill_level", &Range::set_restrict_to_fill_level },
    { "set_show_fill_level",        &Range::set_show_fill_level },
    { "get_adjustment",             &Range::get_adjustment },
    { "set_update_policy",          &Range::set_update_policy },
    { "set_adjustment",             &Range::set_adjustment },
    { "get_inverted",               &Range::get_inverted },
    { "set_inverted",               &Range::set_inverted },
    { "get_update_policy",          &Range::get_update_policy },
    { "get_value",                  &Range::get_value },
    { "set_increments",             &Range::set_increments },
    { "set_range",                  &Range::set_range },
    { "set_value",                  &Range::set_value },
    { "set_lower_stepper_sensitivity",&Range::set_lower_stepper_sensitivity },
    { "get_lower_stepper_sensitivity",&Range::get_lower_stepper_sensitivity },
    { "set_upper_stepper_sensitivity",&Range::set_upper_stepper_sensitivity },
    { "get_upper_stepper_sensitivity",&Range::get_upper_stepper_sensitivity },
#if GTK_CHECK_VERSION( 2, 18, 0 )
    { "get_flippable",              &Range::get_flippable },
    { "set_flippable",              &Range::set_flippable },
#endif
#if GTK_CHECK_VERSION( 2, 20, 0 )
    { "get_min_slider_size",        &Range::get_min_slider_size },
    { "get_range_rect",             &Range::get_range_rect },
    { "get_slider_range",           &Range::get_slider_range },
    { "get_slider_size_fixed",      &Range::get_slider_size_fixed },
    { "set_min_slider_size",        &Range::set_min_slider_size },
    { "set_slider_size_fixed",      &Range::set_slider_size_fixed },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Range, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_Range );
#if GTK_CHECK_VERSION( 2, 16, 0 )
    Gtk::Orientable::clsInit( mod, c_Range );
#endif
}


Range::Range( const Falcon::CoreClass* gen, const GtkRange* range )
    :
    Gtk::CoreGObject( gen, (GObject*) range )
{}


Falcon::CoreObject* Range::factory( const Falcon::CoreClass* gen, void* range, bool )
{
    return new Range( gen, (GtkRange*) range );
}


/*#
    @class GtkRange
    @brief Base class for widgets which visualize an adjustment

    GtkRange is the common base class for widgets which visualize an adjustment,
    e.g scales or scrollbars.

    Apart from signals for monitoring the parameters of the adjustment, GtkRange
    provides properties and methods for influencing the sensitivity of the
    "steppers". It also provides properties and methods for setting a "fill level"
    on range widgets. See gtk_range_set_fill_level().
 */


/*#
    @method signal_adjust_bounds GtkRange
    @brief The "adjust-bounds" signal is emitted when the range is adjusted by user action.

    Note the value can be more or less than the range since it depends on the mouse position.
 */
FALCON_FUNC Range::signal_adjust_bounds( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "adjust_bounds", (void*) &Range::on_adjust_bounds, vm );
}


void Range::on_adjust_bounds( GtkRange* obj, gdouble arg1, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "adjust_bounds", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_adjust_bounds", it ) )
            {
                printf(
                "[GtkRange::on_adjust_bounds] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( arg1 );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_change_value GtkRange
    @brief The change-value signal is emitted when a scroll action is performed on a range.

    It allows an application to determine the type of scroll event that occurred
    and the resultant new value. The application can handle the event itself and
    return TRUE to prevent further processing. Or, by returning FALSE, it can
    pass the event to other handlers until the default GTK+ handler is reached.

    The value parameter is unrounded. An application that overrides the change-value
    signal is responsible for clamping the value to the desired number of decimal
    digits; the default GTK+ handler clamps the value based on range->round_digits.

    It is not possible to use delayed update policies in an overridden change-value handler.
 */
FALCON_FUNC Range::signal_change_value( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "change_value", (void*) &Range::on_change_value, vm );
}


gboolean Range::on_change_value( GtkRange* obj, GtkScrollType scroll, gdouble value, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "change_value", false );

    if ( !cs || cs->empty() )
        return FALSE; // pass event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_change_value", it ) )
            {
                printf(
                "[GtkRange::on_change_value] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        vm->pushParam( (int64) scroll );
        vm->pushParam( value );
        vm->callItem( it, 2 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkRange::on_change_value] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // pass event
}


/*#
    @method signal_move_slider GtkRange
    @brief Virtual function that moves the slider. Used for keybindings.
 */
FALCON_FUNC Range::signal_move_slider( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "move_slider", (void*) &Range::on_move_slider, vm );
}


void Range::on_move_slider( GtkRange* obj, GtkScrollType step, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "move_slider", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_move_slider", it ) )
            {
                printf(
                "[GtkRange::on_move_slider] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) step );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method value_changed GtkRange
    @brief Emitted when the range value changes.
 */
FALCON_FUNC Range::signal_value_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "value_changed", (void*) &Range::on_value_changed, vm );
}


void Range::on_value_changed( GtkRange* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "value_changed", "on_value_changed", (VMachine*)_vm );
}


/*#
    @method get_fill_level GtkRange
    @brief Gets the current position of the fill level indicator.
    @return The current fill level
 */
FALCON_FUNC Range::get_fill_level( VMARG )
{
    NO_ARGS
    vm->retval( gtk_range_get_fill_level( GET_RANGE( vm->self() ) ) );
}


/*#
    @method get_restrict_to_fill_level GtkRange
    @brief Gets whether the range is restricted to the fill level.
    @return TRUE if range is restricted to the fill level.
 */
FALCON_FUNC Range::get_restrict_to_fill_level( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_range_get_restrict_to_fill_level( GET_RANGE( vm->self() ) ) );
}


/*#
    @method get_show_fill_level GtkRange
    @brief Gets whether the range displays the fill level graphically.
    @return TRUE if range shows the fill level.
 */
FALCON_FUNC Range::get_show_fill_level( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_range_get_show_fill_level( GET_RANGE( vm->self() ) ) );
}


/*#
    @method set_fill_level GtkRange
    @brief Set the new position of the fill level indicator.
    @param fill_level the new position of the fill level indicator

    The "fill level" is probably best described by its most prominent use case,
    which is an indicator for the amount of pre-buffering in a streaming media
    player. In that use case, the value of the range would indicate the current
    play position, and the fill level would be the position up to which the
    file/stream has been downloaded.

    This amount of prebuffering can be displayed on the range's trough and is
    themeable separately from the trough. To enable fill level display, use
    gtk_range_set_show_fill_level(). The range defaults to not showing the fill level.

    Additionally, it's possible to restrict the range's slider position to values
    which are smaller than the fill level. This is controller by
    gtk_range_set_restrict_to_fill_level() and is by default enabled.
 */
FALCON_FUNC Range::set_fill_level( VMARG )
{
    Item* i_lvl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lvl || !i_lvl->isOrdinal() )
        throw_inv_params( "O" );
#endif
    gtk_range_set_fill_level( GET_RANGE( vm->self() ), i_lvl->forceNumeric() );
}


/*#
    @method set_restrict_to_fill_level GtkRange
    @brief Sets whether the slider is restricted to the fill level.
    @param restrict_to_fill_level Whether the fill level restricts slider movement.
 */
FALCON_FUNC Range::set_restrict_to_fill_level( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_range_set_restrict_to_fill_level( GET_RANGE( vm->self() ),
                                          i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_show_fill_level GtkRange
    @brief Sets whether a graphical fill level is show on the trough.
    @param show_fill_level Whether a fill level indicator graphics is shown.
 */
FALCON_FUNC Range::set_show_fill_level( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_range_set_show_fill_level( GET_RANGE( vm->self() ),
                                   i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_adjustment GtkRange
    @brief Get the GtkAdjustment which is the "model" object for GtkRange.
    @return a GtkAdjustment
 */
FALCON_FUNC Range::get_adjustment( VMARG )
{
    NO_ARGS
    vm->retval( new Gtk::Adjustment( vm->findWKI( "GtkAdjustment" )->asClass(),
                                     gtk_range_get_adjustment( GET_RANGE( vm->self() ) ) ) );
}


/*#
    @method set_update_policy GtkRange
    @brief Sets the update policy for the range.
    @param policy update policy (GtkUpdateType).

    GTK_UPDATE_CONTINUOUS means that anytime the range slider is moved, the
    range value will change and the value_changed signal will be emitted.
    GTK_UPDATE_DELAYED means that the value will be updated after a brief
    timeout where no slider motion occurs, so updates are spaced by a short
    time rather than continuous. GTK_UPDATE_DISCONTINUOUS means that the value
    will only be updated when the user releases the button and ends the slider
    drag operation.
 */
FALCON_FUNC Range::set_update_policy( VMARG )
{
    Item* i_pol = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pol || !i_pol->isInteger() )
        throw_inv_params( "GtkUpdateType" );
#endif
    gtk_range_set_update_policy( GET_RANGE( vm->self() ),
                                 (GtkUpdateType) i_pol->asInteger() );
}


/*#
    @method set_adjustment GtkRange
    @brief Sets the adjustment to be used as the "model" object for this range widget.
    @param adjustment a GtkAdjustment

    The adjustment indicates the current range value, the minimum and maximum
    range values, the step/page increments used for keybindings and scrolling,
    and the page size. The page size is normally 0 for GtkScale and nonzero
    for GtkScrollbar, and indicates the size of the visible area of the widget
    being scrolled. The page size affects the size of the scrollbar slider.
 */
FALCON_FUNC Range::set_adjustment( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !i_adj->isObject() || !IS_DERIVED( i_adj, GtkAdjustment ) )
        throw_inv_params( "GtkAdjustment" );
#endif
    gtk_range_set_adjustment( GET_RANGE( vm->self() ), GET_ADJUSTMENT( *i_adj ) );
}


/*#
    @method get_inverted GtkRange
    @brief Gets the value set by gtk_range_set_inverted().
    @return TRUE if the range is inverted
 */
FALCON_FUNC Range::get_inverted( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_range_get_inverted( GET_RANGE( vm->self() ) ) );
}


/*#
    @method set_inverted GtkRange
    @brief Ranges normally move from lower to higher values as the slider moves from top to bottom or left to right.
    @param setting TRUE to invert the range

    Inverted ranges have higher values at the top or on the right rather than
    on the bottom or left.
 */
FALCON_FUNC Range::set_inverted( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_range_set_inverted( GET_RANGE( vm->self() ), i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_update_policy GtkRange
    @brief Gets the update policy of range.
    @return The current update policy (GtkUpdateType).
 */
FALCON_FUNC Range::get_update_policy( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_range_get_update_policy( GET_RANGE( vm->self() ) ) );
}


/*#
    @method get_value GtkRange
    @brief Gets the current value of the range.
    @return current value of the range.
 */
FALCON_FUNC Range::get_value( VMARG )
{
    NO_ARGS
    vm->retval( gtk_range_get_value( GET_RANGE( vm->self() ) ) );
}


/*#
    @method set_increments GtkRange
    @brief Sets the step and page sizes for the range.
    @param step step size
    @param page page size

    The step size is used when the user clicks the GtkScrollbar arrows or moves
    GtkScale via arrow keys. The page size is used for example when moving via
    Page Up or Page Down keys.
 */
FALCON_FUNC Range::set_increments( VMARG )
{
    Item* i_step = vm->param( 0 );
    Item* i_page = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_step || !i_step->isOrdinal()
        || !i_page || !i_page->isOrdinal() )
        throw_inv_params( "O,O" );
#endif
    gtk_range_set_increments( GET_RANGE( vm->self() ),
                              i_step->forceNumeric(), i_page->forceNumeric() );
}


/*#
    @method set_range GtkRange
    @brief Sets the allowable values in the GtkRange, and clamps the range value to be between min and max.
    @param min minimum range value
    @param max maximum range value

    If the range has a non-zero page size, it is clamped between min and max - page-size.
 */
FALCON_FUNC Range::set_range( VMARG )
{
    Item* i_min = vm->param( 0 );
    Item* i_max = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_min || !i_min->isOrdinal()
        || !i_max || !i_max->isOrdinal() )
        throw_inv_params( "O,O" );
#endif
    gtk_range_set_increments( GET_RANGE( vm->self() ),
                              i_min->forceNumeric(), i_max->forceNumeric() );
}


/*#
    @method set_value GtkRange
    @brief Sets the current value of the range
    @param value new value of the range

    If the value is outside the minimum or maximum range values, it will be
    clamped to fit inside them. The range emits the "value-changed" signal
    if the value changes.
 */
FALCON_FUNC Range::set_value( VMARG )
{
    Item* i_value = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_value || !i_value->isOrdinal() )
        throw_inv_params( "O" );
#endif
    gtk_range_set_value( GET_RANGE( vm->self() ), i_value->forceNumeric() );
}


/*#
    @method set_lower_stepper_sensitivity GtkRange
    @brief Sets the sensitivity policy for the stepper that points to the 'lower' end of the GtkRange's adjustment.
    @param sensitivity the lower stepper's sensitivity policy (GtkSensitivityType).
 */
FALCON_FUNC Range::set_lower_stepper_sensitivity( VMARG )
{
    Item* i_sens = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sens || !i_sens->isInteger() )
        throw_inv_params( "GtkSensitivityType" );
#endif
    gtk_range_set_lower_stepper_sensitivity( GET_RANGE( vm->self() ),
                                             (GtkSensitivityType) i_sens->asInteger() );
}


/*#
    @method get_lower_stepper_sensitivity GtkRange
    @brief Gets the sensitivity policy for the stepper that points to the 'lower' end of the GtkRange's adjustment.
    @return The lower stepper's sensitivity policy.
 */
FALCON_FUNC Range::get_lower_stepper_sensitivity( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_range_get_lower_stepper_sensitivity( GET_RANGE( vm->self() ) ) );
}


/*#
    @method set_upper_stepper_sensitivity GtkRange
    @brief Sets the sensitivity policy for the stepper that points to the 'upper' end of the GtkRange's adjustment.
    @param sensitivity the upper stepper's sensitivity policy.
 */
FALCON_FUNC Range::set_upper_stepper_sensitivity( VMARG )
{
    Item* i_sens = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sens || !i_sens->isInteger() )
        throw_inv_params( "GtkSensitivityType" );
#endif
    gtk_range_set_upper_stepper_sensitivity( GET_RANGE( vm->self() ),
                                             (GtkSensitivityType) i_sens->asInteger() );
}


/*#
    @method get_upper_stepper_sensitivity GtkRange
    @brief Gets the sensitivity policy for the stepper that points to the 'upper' end of the GtkRange's adjustment.
    @return The upper stepper's sensitivity policy.
 */
FALCON_FUNC Range::get_upper_stepper_sensitivity( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_range_get_lower_stepper_sensitivity( GET_RANGE( vm->self() ) ) );
}


#if GTK_CHECK_VERSION( 2, 18, 0 )
/*#
    @method get_flippable GtkRange
    @brief Gets the value set by gtk_range_set_flippable().
    @return TRUE if the range is flippable
 */
FALCON_FUNC Range::get_flippable( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_range_get_flippable( GET_RANGE( vm->self() ) ) );
}


/*#
    @method set_flippable GtkRange
    @brief If a range is flippable, it will switch its direction if it is horizontal and its direction is GTK_TEXT_DIR_RTL.
    @param flippable TRUE to make the range flippable
 */
FALCON_FUNC Range::set_flippable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_range_set_flippable( GET_RANGE( vm->self() ), i_bool->asBoolean() ? TRUE : FALSE );
}
#endif // GTK_CHECK_VERSION( 2, 18, 0 )


#if GTK_CHECK_VERSION( 2, 20, 0 )
/*#
    @method get_min_slider_size GtkRange
    @brief This function is useful mainly for GtkRange subclasses.
    @return The minimum size of the range's slider.
 */
FALCON_FUNC Range::get_min_slider_size( VMARG )
{
    NO_ARGS
    vm->retval( gtk_range_get_min_slider_size( GET_RANGE( vm->self() ) ) );
}


/*#
    @method get_range_rect GtkRange
    @brief This function returns the area that contains the range's trough and its steppers, in widget->window coordinates.
    @return the range rectangle (GdkRectangle).

    This function is useful mainly for GtkRange subclasses.
 */
FALCON_FUNC Range::get_range_rect( VMARG )
{
    NO_ARGS
    GdkRectangle rect;
    gtk_range_get_range_rect( GET_RANGE( vm->self() ), &rect );
    vm->retval( new Gdk::Rectangle( vm->findWKI( "GdkRectangle" )->asClass(), &rect ) );
}


/*#
    @method get_slider_range GtkRange
    @brief This function returns sliders range along the long dimension, in widget->window coordinates.
    @return Array [ slider start, slider end ]

    This function is useful mainly for GtkRange subclasses.
 */
FALCON_FUNC Range::get_slider_range( VMARG )
{
    NO_ARGS
    gint st, nd;
    gtk_range_get_slider_range( GET_RANGE( vm->self() ), &st, &nd );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( st );
    arr->append( nd );
    vm->retval( arr );
}


/*#
    @method get_slider_size_fixed GtkRange
    @brief This function is useful mainly for GtkRange subclasses.
    @return whether the range's slider has a fixed size.
 */
FALCON_FUNC Range::get_slider_size_fixed( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_range_get_slider_size_fixed( GET_RANGE( vm->self() ) ) );
}


/*#
    @method set_min_slider_size GtkRange
    @brief Sets the minimum size of the range's slider.
    @param min_size The slider's minimum size

    This function is useful mainly for GtkRange subclasses.
 */
FALCON_FUNC Range::set_min_slider_size( VMARG )
{
    Item* i_sz = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sz || !i_sz->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_range_set_min_slider_size( GET_RANGE( vm->self() ), i_sz->asInteger() );
}


/*#
    @method set_slider_size_fixed GtkRange
    @brief Sets whether the range's slider has a fixed size, or a size that depends on it's adjustment's page size.
    @param size_fixed TRUE to make the slider size constant

    This function is useful mainly for GtkRange subclasses.
 */
FALCON_FUNC Range::set_slider_size_fixed( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_range_set_slider_size_fixed( GET_RANGE( vm->self() ), i_bool->asBoolean() ? TRUE : FALSE );
}
#endif // GTK_CHECK_VERSION( 2, 20, 0 )


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
