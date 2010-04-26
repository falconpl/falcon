/**
 *  \file gtk_ProgressBar.cpp
 */

#include "gtk_ProgressBar.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ProgressBar::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ProgressBar = mod->addClass( "GtkProgressBar", &ProgressBar::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkProgress" ) );
    c_ProgressBar->getClassDef()->addInheritance( in );

    c_ProgressBar->getClassDef()->factory( &ProgressBar::factory );

    Gtk::MethodTab methods[] =
    {
    { "pulse",              &ProgressBar::pulse },
    { "set_text",           &ProgressBar::set_text },
    { "set_fraction",       &ProgressBar::set_fraction },
    { "set_pulse_step",     &ProgressBar::set_pulse_step },
    { "set_orientation",    &ProgressBar::set_orientation },
    { "set_ellipsize",      &ProgressBar::set_ellipsize },
    { "get_text",           &ProgressBar::get_text },
    { "get_fraction",       &ProgressBar::get_fraction },
    { "get_pulse_step",     &ProgressBar::get_pulse_step },
    { "get_orientation",    &ProgressBar::get_orientation },
    { "get_ellipsize",      &ProgressBar::get_ellipsize },
#if 0
    { "new_with_adjustment",&ProgressBar::new_with_adjustment },
    { "set_bar_style",      &ProgressBar::set_bar_style },
    { "set_discrete_blocks",&ProgressBar::set_discrete_blocks },
    { "set_activity_step",  &ProgressBar::set_activity_step },
    { "set_activity_blocks",&ProgressBar::set_activity_blocks },
    { "update",             &ProgressBar::update },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ProgressBar, meth->name, meth->cb );
}


ProgressBar::ProgressBar( const Falcon::CoreClass* gen, const GtkProgressBar* pbar )
    :
    Gtk::CoreGObject( gen, (GObject*) pbar )
{}


Falcon::CoreObject* ProgressBar::factory( const Falcon::CoreClass* gen, void* pbar, bool )
{
    return new ProgressBar( gen, (GtkProgressBar*) pbar );
}


/*#
    @class GtkProgressBar
    @brief A widget which indicates progress visually

    The GtkProgressBar is typically used to display the progress of a long running
    operation. It provides a visual clue that processing is underway.
    The GtkProgressBar can be used in two different modes: percentage mode and activity mode.

    [...]
 */
FALCON_FUNC ProgressBar::init( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* wdt = gtk_progress_bar_new();
    self->setGObject( (GObject*) wdt );
}


/*#
    @method pulse GtkProgressBar
    @brief Indicates that some progress is made, but you don't know how much. Causes the progress bar to enter "activity mode," where a block bounces back and forth. Each call to gtk_progress_bar_pulse() causes the block to move by a little bit (the amount of movement per pulse is determined by gtk_progress_bar_set_pulse_step()).
 */
FALCON_FUNC ProgressBar::pulse( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_progress_bar_pulse( (GtkProgressBar*)_obj );
}


/*#
    @method set_text GtkProgressBar
    @brief Causes the given text to appear superimposed on the progress bar.
    @param text a UTF-8 string, or nil.
 */
FALCON_FUNC ProgressBar::set_text( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const char* txt = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_progress_bar_set_text( (GtkProgressBar*)_obj, txt );
}


/*#
    @method set_fraction GtkProgressBar
    @brief Causes the progress bar to "fill in" the given fraction of the bar.
    @param fraction fraction of the task that's been completed
    The fraction should be between 0.0 and 1.0, inclusive.
 */
FALCON_FUNC ProgressBar::set_fraction( VMARG )
{
    Item* i_frac = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_frac || i_frac->isNil() || !i_frac->isNumeric() )
        throw_inv_params( "N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_progress_bar_set_fraction( (GtkProgressBar*)_obj, i_frac->asNumeric() );
}


/*#
    @method set_pulse_step GtkProgressBar
    @brief Sets the fraction of total progress bar length to move the bouncing block for each call to gtk_progress_bar_pulse().
    @param fraction fraction between 0.0 and 1.0
 */
FALCON_FUNC ProgressBar::set_pulse_step( VMARG )
{
    Item* i_frac = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_frac || i_frac->isNil() || !i_frac->isNumeric() )
        throw_inv_params( "N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_progress_bar_set_pulse_step( (GtkProgressBar*)_obj, i_frac->asNumeric() );
}


/*#
    @method set_orientation GtkProgressBar
    @brief Causes the progress bar to switch to a different orientation (left-to-right, right-to-left, top-to-bottom, or bottom-to-top).
    @param orientation orientation of the progress bar
 */
FALCON_FUNC ProgressBar::set_orientation( VMARG )
{
    Item* i_ori = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ori || i_ori->isNil() || !i_ori->isInteger() )
        throw_inv_params( "GtkProgressBarOrientation" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_progress_bar_set_orientation(
            (GtkProgressBar*)_obj, (GtkProgressBarOrientation) i_ori->asInteger() );
}


/*#
    @method set_ellipsize GtkProgressBar
    @brief Sets the mode used to ellipsize (add an ellipsis: "...") the text if there is not enough space to render the entire string.
    @param mode a PangoEllipsizeMode
 */
FALCON_FUNC ProgressBar::set_ellipsize( VMARG )
{
    Item* i_mode = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mode || i_mode->isNil() || !i_mode->isInteger() )
        throw_inv_params( "PangoEllipsizeMode" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_progress_bar_set_ellipsize(
            (GtkProgressBar*)_obj, (PangoEllipsizeMode) i_mode->asInteger() );
}


/*#
    @method get_text GtkProgressBar
    @brief Retrieves the text displayed superimposed on the progress bar, if any, otherwise nil.
    @return text, or nil.
 */
FALCON_FUNC ProgressBar::get_text( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* txt = gtk_progress_bar_get_text( (GtkProgressBar*)_obj );
    if ( txt )
        vm->retval( new String( txt ) );
    else
        vm->retnil();
}


/*#
    @method get_fraction GtkProgressBar
    @brief Returns the current fraction of the task that's been completed.
    @return a fraction from 0.0 to 1.0
 */
FALCON_FUNC ProgressBar::get_fraction( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gdouble frac = gtk_progress_bar_get_fraction( (GtkProgressBar*)_obj );
    vm->retval( frac );
}


/*#
    @method get_pulse_step GtkProgressBar
    @brief Retrieves the pulse step set with gtk_progress_bar_set_pulse_step()
    @return a fraction from 0.0 to 1.0
 */
FALCON_FUNC ProgressBar::get_pulse_step( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gdouble frac = gtk_progress_bar_get_pulse_step( (GtkProgressBar*)_obj );
    vm->retval( frac );
}


/*#
    @method get_orientation GtkProgressBar
    @brief Retrieves the current progress bar orientation.
    @return orientation of the progress bar
 */
FALCON_FUNC ProgressBar::get_orientation( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkProgressBarOrientation ori = gtk_progress_bar_get_orientation( (GtkProgressBar*)_obj );
    vm->retval( (int64) ori );
}


/*#
    @method get_ellipsize GtkProgressBar
    @brief Returns the ellipsizing position of the progressbar.
    @return PangoEllipsizeMode
 */
FALCON_FUNC ProgressBar::get_ellipsize( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    PangoEllipsizeMode mode = gtk_progress_bar_get_ellipsize( (GtkProgressBar*)_obj );
    vm->retval( (int64) mode );
}


#if 0
FALCON_FUNC ProgressBar::new_with_adjustment( VMARG );
FALCON_FUNC ProgressBar::set_bar_style( VMARG );
FALCON_FUNC ProgressBar::set_discrete_blocks( VMARG );
FALCON_FUNC ProgressBar::set_activity_step( VMARG );
FALCON_FUNC ProgressBar::set_activity_blocks( VMARG );
FALCON_FUNC ProgressBar::update( VMARG );
#endif


} // Gtk
} // Falcon
